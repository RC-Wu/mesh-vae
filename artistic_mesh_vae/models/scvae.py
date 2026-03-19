from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

import trellis2.modules.sparse as sp
from trellis2.modules.sparse.transformer.blocks import SparseTransformerBlock
from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder, SparseUnetVaeEncoder


def bins_to_normalized_offsets(bin_values: torch.Tensor, num_bins: int) -> torch.Tensor:
    return ((bin_values + 0.5) / float(num_bins)) * 2.0 - 1.0


def logits_to_expected_offsets(logits: torch.Tensor, num_bins: int) -> torch.Tensor:
    bin_range = torch.arange(num_bins, device=logits.device, dtype=logits.dtype)
    expected_bins = (F.softmax(logits, dim=-1) * bin_range).sum(dim=-1)
    return bins_to_normalized_offsets(expected_bins, num_bins)


def normalized_offsets_to_vertices(
    coords_xyz: torch.Tensor,
    normalized_offsets: torch.Tensor,
    max_offset_per_face: torch.Tensor,
    resolution: int,
) -> torch.Tensor:
    voxel_centers = (coords_xyz.float() + 0.5) / float(resolution)
    offsets = normalized_offsets.reshape(-1, 3, 3) * max_offset_per_face[:, None, None]
    return voxel_centers[:, None, :] + offsets


def compute_total_grad_norm(module: nn.Module) -> torch.Tensor:
    total = None
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        grad_norm = parameter.grad.detach().norm(2)
        total = grad_norm.square() if total is None else total + grad_norm.square()
    if total is None:
        device = next(module.parameters()).device
        return torch.zeros((), device=device)
    return total.sqrt()


class TokenTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = max(dim, int(dim * mlp_ratio))
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class FixedTokenCrossAttentionBottleneck(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_tokens: int,
        token_dim: int,
        num_heads: int,
        num_self_attn_blocks: int = 1,
        mlp_ratio: float = 4.0,
        coord_mlp_hidden: int = 128,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        self.site_proj = nn.Linear(input_dim, token_dim)
        self.coord_proj = nn.Sequential(
            nn.Linear(3, coord_mlp_hidden),
            nn.SiLU(),
            nn.Linear(coord_mlp_hidden, token_dim),
        )
        self.token_queries = nn.Parameter(torch.randn(num_tokens, token_dim) * 0.02)
        self.encode_attn = nn.MultiheadAttention(token_dim, num_heads=num_heads, batch_first=True)
        self.token_blocks = nn.ModuleList(
            [TokenTransformerBlock(token_dim, num_heads=num_heads, mlp_ratio=mlp_ratio) for _ in range(num_self_attn_blocks)]
        )
        self.to_mean = nn.Linear(token_dim, token_dim)
        self.to_logvar = nn.Linear(token_dim, token_dim)
        self.decode_attn = nn.MultiheadAttention(token_dim, num_heads=num_heads, batch_first=True)
        self.decode_out = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, latent_dim),
        )

    def _pack_sites(self, sparse_tensor: sp.SparseTensor, resolution: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = sparse_tensor.feats
        coords = sparse_tensor.coords
        batch_index = coords[:, 0].to(torch.long).reshape(-1)
        unique_batch_ids, batch_index_mapped = torch.unique(batch_index, sorted=True, return_inverse=True)
        batch_size = int(unique_batch_ids.numel()) if unique_batch_ids.numel() > 0 else 1
        order = torch.argsort(batch_index_mapped)
        batch_sorted = batch_index_mapped[order]
        counts = torch.bincount(batch_sorted, minlength=batch_size)
        max_sites = int(counts.max().item()) if counts.numel() > 0 else 0

        padded_feats = feats.new_zeros((batch_size, max_sites, feats.shape[-1]))
        padded_coords = feats.new_zeros((batch_size, max_sites, 3))
        key_padding_mask = torch.ones((batch_size, max_sites), dtype=torch.bool, device=feats.device)

        sorted_feats = feats[order]
        sorted_coords = ((coords[order, 1:].float() + 0.5) / float(resolution)).to(feats.dtype)
        offset = 0
        for batch_id, count in enumerate(counts.detach().cpu().tolist()):
            if count == 0:
                continue
            padded_feats[batch_id, :count] = sorted_feats[offset : offset + count]
            padded_coords[batch_id, :count] = sorted_coords[offset : offset + count]
            key_padding_mask[batch_id, :count] = False
            offset += count
        return padded_feats, padded_coords, key_padding_mask, order

    def _unpack_sites(self, decoded: torch.Tensor, key_padding_mask: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        pieces: List[torch.Tensor] = []
        for batch_id in range(decoded.shape[0]):
            valid = ~key_padding_mask[batch_id]
            if valid.any():
                pieces.append(decoded[batch_id, valid])
        decoded_sorted = torch.cat(pieces, dim=0) if pieces else decoded.new_zeros((0, self.latent_dim))
        inverse_order = torch.empty_like(order)
        inverse_order[order] = torch.arange(order.numel(), device=order.device)
        return decoded_sorted[inverse_order]

    def forward(self, sparse_tensor: sp.SparseTensor, resolution: int, sample_posterior: bool) -> Tuple[sp.SparseTensor, torch.Tensor, torch.Tensor]:
        padded_feats, padded_coords, key_padding_mask, order = self._pack_sites(sparse_tensor, resolution=resolution)
        site_inputs = self.site_proj(padded_feats) + self.coord_proj(padded_coords)
        queries = self.token_queries.unsqueeze(0).expand(site_inputs.shape[0], -1, -1)
        tokens, _ = self.encode_attn(queries, site_inputs, site_inputs, key_padding_mask=key_padding_mask, need_weights=False)
        for block in self.token_blocks:
            tokens = block(tokens)
        mean = self.to_mean(tokens)
        logvar = self.to_logvar(tokens)
        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            token_latents = mean + std * torch.randn_like(std)
        else:
            token_latents = mean

        site_queries = self.coord_proj(padded_coords)
        decoded_sites, _ = self.decode_attn(site_queries, token_latents, token_latents, need_weights=False)
        decoded_sites = self.decode_out(decoded_sites)
        latent_feats = self._unpack_sites(decoded_sites, key_padding_mask=key_padding_mask, order=order)
        latent_sparse = sparse_tensor.replace(latent_feats)
        return latent_sparse, mean, logvar


class QuantizedFaceVaeEncoder(SparseUnetVaeEncoder):
    def __init__(
        self,
        in_channels: int,
        model_channels: Sequence[int],
        latent_channels: int,
        num_blocks: Sequence[int],
        block_type: Sequence[str],
        down_block_type: Sequence[str],
        block_args: Sequence[Dict[str, Any]],
        use_fp16: bool = False,
        bottleneck_attention_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            in_channels=in_channels,
            model_channels=list(model_channels),
            latent_channels=latent_channels,
            num_blocks=list(num_blocks),
            block_type=list(block_type),
            down_block_type=list(down_block_type),
            block_args=list(block_args),
            use_fp16=use_fp16,
        )
        self.bottleneck_attention_blocks = nn.ModuleList()
        attention_cfg = dict(bottleneck_attention_cfg or {})
        num_attention_blocks = int(attention_cfg.get("num_blocks", 0))
        if num_attention_blocks > 0:
            channels = int(model_channels[-1])
            for _ in range(num_attention_blocks):
                self.bottleneck_attention_blocks.append(
                    SparseTransformerBlock(
                        channels=channels,
                        num_heads=int(attention_cfg.get("num_heads", 8)),
                        mlp_ratio=float(attention_cfg.get("mlp_ratio", 4.0)),
                        attn_mode=str(attention_cfg.get("attn_mode", "full")),
                        window_size=attention_cfg.get("window_size"),
                        shift_window=tuple(attention_cfg["shift_window"]) if attention_cfg.get("shift_window") else None,
                        use_checkpoint=bool(attention_cfg.get("use_checkpoint", False)),
                        use_rope=bool(attention_cfg.get("use_rope", False)),
                        rope_freq=tuple(attention_cfg.get("rope_freq", (1.0, 10000.0))),
                        qk_rms_norm=bool(attention_cfg.get("qk_rms_norm", False)),
                        qkv_bias=bool(attention_cfg.get("qkv_bias", True)),
                        ln_affine=bool(attention_cfg.get("ln_affine", False)),
                    )
                )

    def forward_features(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = self.input_layer(x)
        h = h.type(self.dtype)
        for res in self.blocks:
            for block in res:
                h = block(h)
        for block in self.bottleneck_attention_blocks:
            h = block(h)
        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        return h

    def posterior_from_features(self, h: sp.SparseTensor, sample_posterior: bool = False, return_raw: bool = False):
        h = self.to_latent(h)

        mean, logvar = h.feats.chunk(2, dim=-1)
        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean
        z = h.replace(z)
        if return_raw:
            return z, mean, logvar
        return z

    def forward(self, x: sp.SparseTensor, sample_posterior: bool = False, return_raw: bool = False):
        h = self.forward_features(x)
        return self.posterior_from_features(h, sample_posterior=sample_posterior, return_raw=return_raw)


class QuantizedFaceVaeDecoder(SparseUnetVaeDecoder):
    def __init__(
        self,
        out_channels: int,
        model_channels: Sequence[int],
        latent_channels: int,
        num_blocks: Sequence[int],
        block_type: Sequence[str],
        up_block_type: Sequence[str],
        block_args: Sequence[Dict[str, Any]],
        use_fp16: bool = False,
        pred_subdiv: bool = True,
        bottleneck_attention_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            out_channels=out_channels,
            model_channels=list(model_channels),
            latent_channels=latent_channels,
            num_blocks=list(num_blocks),
            block_type=list(block_type),
            up_block_type=list(up_block_type),
            block_args=list(block_args),
            use_fp16=use_fp16,
            pred_subdiv=pred_subdiv,
        )
        self.bottleneck_attention_blocks = nn.ModuleList()
        attention_cfg = dict(bottleneck_attention_cfg or {})
        num_attention_blocks = int(attention_cfg.get("num_blocks", 0))
        if num_attention_blocks > 0:
            channels = int(model_channels[0])
            for _ in range(num_attention_blocks):
                self.bottleneck_attention_blocks.append(
                    SparseTransformerBlock(
                        channels=channels,
                        num_heads=int(attention_cfg.get("num_heads", 8)),
                        mlp_ratio=float(attention_cfg.get("mlp_ratio", 4.0)),
                        attn_mode=str(attention_cfg.get("attn_mode", "full")),
                        window_size=attention_cfg.get("window_size"),
                        shift_window=tuple(attention_cfg["shift_window"]) if attention_cfg.get("shift_window") else None,
                        use_checkpoint=bool(attention_cfg.get("use_checkpoint", False)),
                        use_rope=bool(attention_cfg.get("use_rope", False)),
                        rope_freq=tuple(attention_cfg.get("rope_freq", (1.0, 10000.0))),
                        qk_rms_norm=bool(attention_cfg.get("qk_rms_norm", False)),
                        qkv_bias=bool(attention_cfg.get("qkv_bias", True)),
                        ln_affine=bool(attention_cfg.get("ln_affine", False)),
                    )
                )

    def forward(
        self,
        x: sp.SparseTensor,
        guide_subs: Optional[List[sp.SparseTensor]] = None,
        return_subs: bool = False,
    ) -> sp.SparseTensor:
        assert guide_subs is None or self.pred_subdiv == False, "Only decoders with pred_subdiv=False can be used with guide_subs"
        assert return_subs == False or self.pred_subdiv == True, "Only decoders with pred_subdiv=True can be used with return_subs"

        h = self.from_latent(x)
        h = h.type(self.dtype)
        for block in self.bottleneck_attention_blocks:
            h = block(h)
        subs_gt = []
        subs = []
        for i, res in enumerate(self.blocks):
            for j, block in enumerate(res):
                if i < len(self.blocks) - 1 and j == len(res) - 1:
                    if self.pred_subdiv:
                        if self.training:
                            subs_gt.append(h.get_spatial_cache("subdivision"))
                        h, sub = block(h)
                        subs.append(sub)
                    else:
                        h = block(h, subdiv=guide_subs[i] if guide_subs is not None else None)
                else:
                    h = block(h)
        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.output_layer(h)
        if self.training and self.pred_subdiv:
            return h, subs_gt, subs
        if return_subs:
            return h, subs
        return h


class CausalFaceMLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_bins: int,
        mlp_hidden: int,
        num_layers: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_bins = num_bins
        self.vert_embed_dim = min(128, max(32, hidden_dim // 4))
        self.vert_embedding = nn.Linear(3 * num_bins, self.vert_embed_dim)
        self.mlps = nn.ModuleList()
        for step in range(3):
            input_dim = hidden_dim + step * self.vert_embed_dim
            layers: List[nn.Module] = []
            for layer_index in range(num_layers):
                layers.append(nn.Linear(input_dim if layer_index == 0 else mlp_hidden, mlp_hidden))
                layers.append(nn.SiLU())
            layers.append(nn.Linear(mlp_hidden, 3 * num_bins))
            self.mlps.append(nn.Sequential(*layers))

    def _embed_vertex(self, bin_indices: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(bin_indices, self.num_bins).float().reshape(-1, 3 * self.num_bins)
        return self.vert_embedding(one_hot)

    def forward(
        self,
        hidden_feats: torch.Tensor,
        gt_bins: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        use_teacher_forcing = gt_bins is not None
        face_bins = gt_bins.reshape(-1, 3, 3) if gt_bins is not None else None

        logits_v0 = self.mlps[0](hidden_feats).reshape(-1, 3, self.num_bins)
        if use_teacher_forcing:
            v0_bins = face_bins[:, 0]
        else:
            v0_bins = logits_v0.argmax(dim=-1)
        v0_embed = self._embed_vertex(v0_bins)

        logits_v1 = self.mlps[1](torch.cat([hidden_feats, v0_embed], dim=-1)).reshape(-1, 3, self.num_bins)
        if use_teacher_forcing:
            v1_bins = face_bins[:, 1]
        else:
            v1_bins = logits_v1.argmax(dim=-1)
        v1_embed = self._embed_vertex(v1_bins)

        logits_v2 = self.mlps[2](torch.cat([hidden_feats, v0_embed, v1_embed], dim=-1)).reshape(-1, 3, self.num_bins)
        return logits_v0, logits_v1, logits_v2


class RegressionHead(nn.Module):
    def __init__(self, hidden_dim: int, mlp_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, 9),
            nn.Tanh(),
        )

    def forward(self, hidden_feats: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_feats)


class BinaryTopologyHead(nn.Module):
    def __init__(self, hidden_dim: int, head_hidden: int, num_layers: int):
        super().__init__()
        layers: List[nn.Module] = []
        input_dim = hidden_dim
        num_layers = max(1, int(num_layers))
        for layer_index in range(num_layers - 1):
            layers.append(nn.Linear(input_dim if layer_index == 0 else head_hidden, head_hidden))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(input_dim if num_layers == 1 else head_hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, hidden_feats: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_feats).squeeze(-1)


class BinaryRoleHead(nn.Module):
    def __init__(self, hidden_dim: int, head_hidden: int, num_layers: int):
        super().__init__()
        layers: List[nn.Module] = []
        input_dim = hidden_dim
        num_layers = max(1, int(num_layers))
        for layer_index in range(num_layers - 1):
            layers.append(nn.Linear(input_dim if layer_index == 0 else head_hidden, head_hidden))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(input_dim if num_layers == 1 else head_hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, hidden_feats: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_feats).squeeze(-1)


class QuantizedFaceVaeModule(pl.LightningModule):
    def __init__(
        self,
        model_cfg: Dict[str, Any],
        loss_cfg: Dict[str, Any],
        optim_cfg: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters(
            {
                "model_cfg": model_cfg,
                "loss_cfg": loss_cfg,
                "optim_cfg": optim_cfg,
            }
        )
        self.model_cfg = model_cfg
        self.loss_cfg = loss_cfg
        self.optim_cfg = optim_cfg
        self.resolution = int(model_cfg["resolution"])
        self.num_bins = int(model_cfg["num_bins"])
        self.continuous_only = bool(model_cfg.get("continuous_only", False))
        self.consistency_max_pairs = int(loss_cfg.get("consistency_max_pairs", 50000))
        self.predict_subdiv = bool(model_cfg.get("predict_subdiv", True))
        self.lambda_topo_edge = float(loss_cfg.get("lambda_topo_edge", 0.0))
        self.lambda_topo_open_boundary = float(loss_cfg.get("lambda_topo_open_boundary", 0.0))
        self.lambda_structure_token = float(loss_cfg.get("lambda_structure_token", 0.0))
        self.lambda_fill_token = float(loss_cfg.get("lambda_fill_token", 0.0))
        self.topology_head_hidden = int(model_cfg.get("topology_head_hidden", max(64, int(model_cfg["decoder_out_channels"]) // 2)))
        self.topology_head_num_layers = int(model_cfg.get("topology_head_num_layers", 2))
        self.role_head_hidden = int(model_cfg.get("role_head_hidden", self.topology_head_hidden))
        self.role_head_num_layers = int(model_cfg.get("role_head_num_layers", self.topology_head_num_layers))
        token_bottleneck_cfg = dict(model_cfg.get("token_bottleneck") or {})
        behavior_cfg = dict(model_cfg.get("behavior") or {})
        self.train_behavior = self._resolve_behavior(
            behavior_cfg.get("train"),
            sample_posterior=True,
            teacher_forcing=True,
            guided_structure=True,
        )
        self.val_behavior = self._resolve_behavior(
            behavior_cfg.get("val"),
            sample_posterior=False,
            teacher_forcing=False,
            guided_structure=True,
        )

        encoder_channels = list(model_cfg["model_channels"])
        decoder_channels = list(reversed(encoder_channels))
        encoder_num_blocks = list(model_cfg["num_blocks"])
        decoder_num_blocks = list(reversed(encoder_num_blocks))
        block_type = list(model_cfg["block_type"])
        up_block_type = list(model_cfg["up_block_type"])
        down_block_type = list(model_cfg["down_block_type"])
        block_args = list(model_cfg.get("block_args", [{} for _ in encoder_channels]))
        use_fp16 = bool(model_cfg.get("use_fp16", False))
        bottleneck_attention_cfg = dict(model_cfg.get("bottleneck_attention") or {})

        self.encoder = QuantizedFaceVaeEncoder(
            in_channels=int(model_cfg["feature_dim"]),
            model_channels=encoder_channels,
            latent_channels=int(model_cfg["latent_channels"]),
            num_blocks=encoder_num_blocks,
            block_type=block_type,
            down_block_type=down_block_type,
            block_args=block_args,
            use_fp16=use_fp16,
            bottleneck_attention_cfg=bottleneck_attention_cfg,
        )
        self.token_bottleneck = None
        if bool(token_bottleneck_cfg.get("enabled", False)):
            self.token_bottleneck = FixedTokenCrossAttentionBottleneck(
                input_dim=int(encoder_channels[-1]),
                latent_dim=int(model_cfg["latent_channels"]),
                num_tokens=int(token_bottleneck_cfg.get("num_tokens", 128)),
                token_dim=int(token_bottleneck_cfg.get("token_dim", max(int(model_cfg["latent_channels"]), 64))),
                num_heads=int(token_bottleneck_cfg.get("num_heads", 8)),
                num_self_attn_blocks=int(token_bottleneck_cfg.get("num_self_attn_blocks", 1)),
                mlp_ratio=float(token_bottleneck_cfg.get("mlp_ratio", 4.0)),
                coord_mlp_hidden=int(token_bottleneck_cfg.get("coord_mlp_hidden", 128)),
            )
        self.decoder = QuantizedFaceVaeDecoder(
            out_channels=int(model_cfg["decoder_out_channels"]),
            model_channels=decoder_channels,
            latent_channels=int(model_cfg["latent_channels"]),
            num_blocks=decoder_num_blocks,
            block_type=list(reversed(block_type)),
            up_block_type=list(reversed(up_block_type)),
            block_args=list(reversed(block_args)),
            use_fp16=use_fp16,
            pred_subdiv=self.predict_subdiv,
            bottleneck_attention_cfg=bottleneck_attention_cfg,
        )
        self.head = None
        if not self.continuous_only:
            self.head = CausalFaceMLP(
                hidden_dim=int(model_cfg["decoder_out_channels"]),
                num_bins=self.num_bins,
                mlp_hidden=int(model_cfg["head_mlp_hidden"]),
                num_layers=int(model_cfg["head_num_layers"]),
            )
        self.topology_edge_head = None
        self.topology_open_boundary_head = None
        self.structure_token_head = None
        self.fill_token_head = None
        if self.lambda_topo_edge > 0.0:
            self.topology_edge_head = BinaryTopologyHead(
                hidden_dim=int(model_cfg["decoder_out_channels"]),
                head_hidden=self.topology_head_hidden,
                num_layers=self.topology_head_num_layers,
            )
        if self.lambda_topo_open_boundary > 0.0:
            self.topology_open_boundary_head = BinaryTopologyHead(
                hidden_dim=int(model_cfg["decoder_out_channels"]),
                head_hidden=self.topology_head_hidden,
                num_layers=self.topology_head_num_layers,
            )
        if self.lambda_structure_token > 0.0:
            self.structure_token_head = BinaryRoleHead(
                hidden_dim=int(model_cfg["decoder_out_channels"]),
                head_hidden=self.role_head_hidden,
                num_layers=self.role_head_num_layers,
            )
        if self.lambda_fill_token > 0.0:
            self.fill_token_head = BinaryRoleHead(
                hidden_dim=int(model_cfg["decoder_out_channels"]),
                head_hidden=self.role_head_hidden,
                num_layers=self.role_head_num_layers,
            )
        self.regression_head = None
        if float(loss_cfg.get("lambda_direct_regression", 0.0)) > 0:
            self.regression_head = RegressionHead(
                hidden_dim=int(model_cfg["decoder_out_channels"]),
                mlp_hidden=int(model_cfg.get("regression_head_hidden", model_cfg["decoder_out_channels"])),
            )
        if self.continuous_only and self.regression_head is None:
            raise ValueError("continuous_only=true requires loss.lambda_direct_regression > 0 to enable the regression head")

    @staticmethod
    def _resolve_behavior(
        override: Optional[Dict[str, Any]],
        *,
        sample_posterior: bool,
        teacher_forcing: bool,
        guided_structure: bool,
    ) -> Dict[str, bool]:
        payload = dict(override or {})
        return {
            "sample_posterior": bool(payload.get("sample_posterior", sample_posterior)),
            "teacher_forcing": bool(payload.get("teacher_forcing", teacher_forcing)),
            "guided_structure": bool(payload.get("guided_structure", guided_structure)),
        }

    def configure_optimizers(self):
        betas = tuple(self.optim_cfg.get("betas", (0.9, 0.95)))
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.optim_cfg["lr"]),
            weight_decay=float(self.optim_cfg.get("weight_decay", 0.0)),
            betas=betas,
        )
        return optimizer

    def on_before_optimizer_step(self, optimizer):
        grad_norm = compute_total_grad_norm(self)
        self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False, prog_bar=False)

    def _build_sparse_input(self, batch: Dict[str, Any]) -> sp.SparseTensor:
        return sp.SparseTensor(feats=batch["feats"], coords=batch["coords"])

    def _resolve_topology_targets(
        self,
        batch: Dict[str, Any],
        device: torch.device,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        stacked = batch.get("topology_flags")
        if stacked is None:
            stacked = batch.get("topo_flags")
        if stacked is not None:
            if not torch.is_tensor(stacked):
                stacked = torch.as_tensor(stacked)
            if stacked.ndim < 2 or stacked.shape[-1] < 2:
                raise ValueError(
                    "topology supervision expects `topology_flags` or `topo_flags` with shape [..., 2]"
                )
            edge_targets = stacked[..., 0]
            open_boundary_targets = stacked[..., 1]
        else:
            edge_targets = batch.get("is_edge_or_vertex")
            open_boundary_targets = batch.get("is_open_boundary_edge")
            if edge_targets is None and open_boundary_targets is None:
                return None
            if edge_targets is None or open_boundary_targets is None:
                raise ValueError(
                    "topology supervision expects both `is_edge_or_vertex` and `is_open_boundary_edge` "
                    "when topology losses are enabled"
                )

        edge_targets = edge_targets.to(device=device).reshape(-1)
        open_boundary_targets = open_boundary_targets.to(device=device).reshape(-1)
        return edge_targets, open_boundary_targets

    def _resolve_role_targets(
        self,
        batch: Dict[str, Any],
        device: torch.device,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        stacked = batch.get("token_role_flags")
        if stacked is None:
            stacked = batch.get("token_flags")
        if stacked is not None:
            if not torch.is_tensor(stacked):
                stacked = torch.as_tensor(stacked)
            if stacked.ndim < 2 or stacked.shape[-1] < 2:
                raise ValueError(
                    "role supervision expects `token_role_flags` or `token_flags` with shape [..., 2]"
                )
            structure_targets = stacked[..., 0]
            fill_targets = stacked[..., 1]
        else:
            structure_targets = batch.get("structure_token")
            if structure_targets is None:
                structure_targets = batch.get("is_structure_token")
            fill_targets = batch.get("fill_token")
            if fill_targets is None:
                fill_targets = batch.get("is_fill_token")
            if structure_targets is None and fill_targets is None:
                return None
            if structure_targets is None or fill_targets is None:
                raise ValueError(
                    "role supervision expects both `structure_token` and `fill_token` "
                    "when structure/fill losses are enabled"
                )

        structure_targets = structure_targets.to(device=device).reshape(-1)
        fill_targets = fill_targets.to(device=device).reshape(-1)
        return structure_targets, fill_targets

    @staticmethod
    def _binary_head_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        if logits.numel() == 0 or targets.numel() == 0:
            zeros = logits.new_zeros(())
            return {
                "loss": zeros,
                "acc": zeros,
                "target_pos_rate": zeros,
                "pred_pos_rate": zeros,
            }
        targets_float = targets.float()
        probs = torch.sigmoid(logits)
        preds = probs >= 0.5
        return {
            "loss": F.binary_cross_entropy_with_logits(logits, targets_float),
            "acc": preds.eq(targets.bool()).float().mean(),
            "target_pos_rate": targets_float.mean(),
            "pred_pos_rate": preds.float().mean(),
        }

    def _decode(
        self,
        latent_sparse: sp.SparseTensor,
        guided_structure: bool,
    ) -> Tuple[sp.SparseTensor, List[torch.Tensor], List[Any]]:
        if guided_structure and self.predict_subdiv:
            previous_mode = self.decoder.training
            self.decoder.train(True)
            pred_sparse, subs_gt, subs = self.decoder(latent_sparse)
            self.decoder.train(previous_mode)
            return pred_sparse, subs_gt, subs

        decoded = self.decoder(latent_sparse, return_subs=self.predict_subdiv)
        if self.predict_subdiv:
            pred_sparse, subs = decoded
        else:
            pred_sparse, subs = decoded, []
        return pred_sparse, [], subs

    def _forward_impl(
        self,
        batch: Dict[str, Any],
        sample_posterior: bool,
        teacher_forcing: bool,
        guided_structure: bool,
    ) -> Dict[str, Any]:
        sparse_input = self._build_sparse_input(batch)
        if self.token_bottleneck is not None:
            encoder_features = self.encoder.forward_features(sparse_input)
            latent_sparse, mean, logvar = self.token_bottleneck(
                encoder_features,
                resolution=self.resolution,
                sample_posterior=sample_posterior,
            )
        else:
            latent_sparse, mean, logvar = self.encoder(sparse_input, sample_posterior=sample_posterior, return_raw=True)
        pred_sparse, subs_gt, subs = self._decode(latent_sparse, guided_structure=guided_structure)
        hidden_feats = pred_sparse.feats
        gt_bins = batch["bin_indices"] if teacher_forcing else None
        if self.head is not None:
            logits_v0, logits_v1, logits_v2 = self.head(hidden_feats, gt_bins=gt_bins)
        else:
            logits_v0 = logits_v1 = logits_v2 = None
        topology_logits_edge = self.topology_edge_head(hidden_feats) if self.topology_edge_head is not None else None
        topology_logits_open_boundary = (
            self.topology_open_boundary_head(hidden_feats) if self.topology_open_boundary_head is not None else None
        )
        structure_token_logits = self.structure_token_head(hidden_feats) if self.structure_token_head is not None else None
        fill_token_logits = self.fill_token_head(hidden_feats) if self.fill_token_head is not None else None
        direct_regression = self.regression_head(hidden_feats) if self.regression_head is not None else None
        return {
            "pred_sparse": pred_sparse,
            "logits_v0": logits_v0,
            "logits_v1": logits_v1,
            "logits_v2": logits_v2,
            "topology_logits_edge": topology_logits_edge,
            "topology_logits_open_boundary": topology_logits_open_boundary,
            "structure_token_logits": structure_token_logits,
            "fill_token_logits": fill_token_logits,
            "direct_regression": direct_regression,
            "subs_gt": subs_gt,
            "subs": subs,
            "mean": mean,
            "logvar": logvar,
        }

    def _compose_pred_bins(self, outputs: Dict[str, Any]) -> torch.Tensor:
        if outputs["logits_v0"] is None:
            if outputs["direct_regression"] is None:
                raise ValueError("cannot compose predicted bins without discrete logits or direct regression outputs")
            normalized01 = (outputs["direct_regression"].clamp(-1.0, 1.0) + 1.0) * 0.5
            return torch.clamp(torch.floor(normalized01 * float(self.num_bins)).long(), 0, self.num_bins - 1)
        return torch.cat(
            [
                outputs["logits_v0"].argmax(dim=-1),
                outputs["logits_v1"].argmax(dim=-1),
                outputs["logits_v2"].argmax(dim=-1),
            ],
            dim=-1,
        )

    def _compose_soft_offsets(self, outputs: Dict[str, Any]) -> torch.Tensor:
        if outputs["logits_v0"] is None:
            if outputs["direct_regression"] is None:
                raise ValueError("cannot compose soft offsets without discrete logits or direct regression outputs")
            return outputs["direct_regression"]
        return torch.cat(
            [
                logits_to_expected_offsets(outputs["logits_v0"], self.num_bins),
                logits_to_expected_offsets(outputs["logits_v1"], self.num_bins),
                logits_to_expected_offsets(outputs["logits_v2"], self.num_bins),
            ],
            dim=-1,
        )

    def _compute_consistency(self, batch: Dict[str, Any], normalized_offsets: torch.Tensor) -> torch.Tensor:
        if batch["adj_fi"].numel() == 0:
            return normalized_offsets.new_zeros(())

        face_vertices = normalized_offsets_to_vertices(
            coords_xyz=batch["coords_xyz"].to(normalized_offsets.device),
            normalized_offsets=normalized_offsets,
            max_offset_per_face=batch["max_offset_per_face"].to(normalized_offsets.device),
            resolution=self.resolution,
        )
        max_pairs = min(self.consistency_max_pairs, int(batch["adj_fi"].numel()))
        fi = batch["adj_fi"][:max_pairs].to(normalized_offsets.device)
        fj = batch["adj_fj"][:max_pairs].to(normalized_offsets.device)
        vi = batch["adj_vi"][:max_pairs].to(normalized_offsets.device)
        vj = batch["adj_vj"][:max_pairs].to(normalized_offsets.device)
        lhs = face_vertices[fi, vi]
        rhs = face_vertices[fj, vj]
        return F.mse_loss(lhs, rhs)

    def _compute_loss_terms(self, batch: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        gt_bins = batch["bin_indices"]
        gt_offsets = batch["gt_offsets"]

        if outputs["logits_v0"] is not None:
            ce_v0 = F.cross_entropy(outputs["logits_v0"].reshape(-1, self.num_bins), gt_bins[:, 0:3].reshape(-1))
            ce_v1 = F.cross_entropy(outputs["logits_v1"].reshape(-1, self.num_bins), gt_bins[:, 3:6].reshape(-1))
            ce_v2 = F.cross_entropy(outputs["logits_v2"].reshape(-1, self.num_bins), gt_bins[:, 6:9].reshape(-1))
            ce_recon = (ce_v0 + ce_v1 + ce_v2) / 3.0
        else:
            ce_v0 = gt_offsets.new_zeros(())
            ce_v1 = gt_offsets.new_zeros(())
            ce_v2 = gt_offsets.new_zeros(())
            ce_recon = gt_offsets.new_zeros(())

        subdiv_terms = [
            F.binary_cross_entropy_with_logits(sub.feats, sub_gt.float())
            for sub_gt, sub in zip(outputs["subs_gt"], outputs["subs"])
            if sub_gt is not None
        ]
        subdiv_loss = torch.stack(subdiv_terms).mean() if subdiv_terms else gt_offsets.new_zeros(())

        mean = outputs["mean"]
        logvar = outputs["logvar"]
        kl = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1.0)

        pred_bins = self._compose_pred_bins(outputs)
        pred_offsets_soft = self._compose_soft_offsets(outputs)
        pred_offsets_argmax = bins_to_normalized_offsets(pred_bins.float(), self.num_bins)
        aux_regression = F.smooth_l1_loss(pred_offsets_soft, gt_offsets)
        direct_regression = (
            F.smooth_l1_loss(outputs["direct_regression"], gt_offsets)
            if outputs["direct_regression"] is not None
            else gt_offsets.new_zeros(())
        )
        consistency_source = outputs["direct_regression"] if outputs["direct_regression"] is not None else pred_offsets_soft
        consistency = self._compute_consistency(batch, consistency_source)

        topology_targets = None
        if self.topology_edge_head is not None or self.topology_open_boundary_head is not None:
            topology_targets = self._resolve_topology_targets(batch, device=gt_offsets.device)
            if topology_targets is None:
                raise ValueError(
                    "topology losses are enabled but the batch does not contain topology supervision. "
                    "Expected `topology_flags`, `topo_flags`, or the pair `is_edge_or_vertex` / "
                    "`is_open_boundary_edge`."
                )
        topology_edge_loss = gt_offsets.new_zeros(())
        topology_open_boundary_loss = gt_offsets.new_zeros(())
        topology_edge_acc = gt_offsets.new_zeros(())
        topology_open_boundary_acc = gt_offsets.new_zeros(())
        topology_edge_target_pos_rate = gt_offsets.new_zeros(())
        topology_open_boundary_target_pos_rate = gt_offsets.new_zeros(())
        topology_edge_pred_pos_rate = gt_offsets.new_zeros(())
        topology_open_boundary_pred_pos_rate = gt_offsets.new_zeros(())
        if topology_targets is not None:
            edge_targets, open_boundary_targets = topology_targets
            if self.topology_edge_head is not None:
                if edge_targets.shape[0] != outputs["topology_logits_edge"].shape[0]:
                    raise ValueError(
                        "topology edge targets length does not match decoder token count: "
                        f"{edge_targets.shape[0]} vs {outputs['topology_logits_edge'].shape[0]}"
                    )
                edge_metrics = self._binary_head_metrics(outputs["topology_logits_edge"], edge_targets)
                topology_edge_loss = edge_metrics["loss"]
                topology_edge_acc = edge_metrics["acc"]
                topology_edge_target_pos_rate = edge_metrics["target_pos_rate"]
                topology_edge_pred_pos_rate = edge_metrics["pred_pos_rate"]
            if self.topology_open_boundary_head is not None:
                if open_boundary_targets.shape[0] != outputs["topology_logits_open_boundary"].shape[0]:
                    raise ValueError(
                        "topology open-boundary targets length does not match decoder token count: "
                        f"{open_boundary_targets.shape[0]} vs {outputs['topology_logits_open_boundary'].shape[0]}"
                    )
                open_boundary_metrics = self._binary_head_metrics(
                    outputs["topology_logits_open_boundary"],
                    open_boundary_targets,
                )
                topology_open_boundary_loss = open_boundary_metrics["loss"]
                topology_open_boundary_acc = open_boundary_metrics["acc"]
                topology_open_boundary_target_pos_rate = open_boundary_metrics["target_pos_rate"]
                topology_open_boundary_pred_pos_rate = open_boundary_metrics["pred_pos_rate"]

        role_targets = None
        if self.structure_token_head is not None or self.fill_token_head is not None:
            role_targets = self._resolve_role_targets(batch, device=gt_offsets.device)
            if role_targets is None:
                raise ValueError(
                    "structure/fill losses are enabled but the batch does not contain role supervision. "
                    "Expected `token_role_flags`, `token_flags`, or the pair `structure_token` / `fill_token`."
                )
        structure_token_loss = gt_offsets.new_zeros(())
        fill_token_loss = gt_offsets.new_zeros(())
        structure_token_acc = gt_offsets.new_zeros(())
        fill_token_acc = gt_offsets.new_zeros(())
        structure_token_target_pos_rate = gt_offsets.new_zeros(())
        fill_token_target_pos_rate = gt_offsets.new_zeros(())
        structure_token_pred_pos_rate = gt_offsets.new_zeros(())
        fill_token_pred_pos_rate = gt_offsets.new_zeros(())
        if role_targets is not None:
            structure_targets, fill_targets = role_targets
            if self.structure_token_head is not None:
                if structure_targets.shape[0] != outputs["structure_token_logits"].shape[0]:
                    raise ValueError(
                        "structure-token targets length does not match decoder token count: "
                        f"{structure_targets.shape[0]} vs {outputs['structure_token_logits'].shape[0]}"
                    )
                structure_metrics = self._binary_head_metrics(outputs["structure_token_logits"], structure_targets)
                structure_token_loss = structure_metrics["loss"]
                structure_token_acc = structure_metrics["acc"]
                structure_token_target_pos_rate = structure_metrics["target_pos_rate"]
                structure_token_pred_pos_rate = structure_metrics["pred_pos_rate"]
            if self.fill_token_head is not None:
                if fill_targets.shape[0] != outputs["fill_token_logits"].shape[0]:
                    raise ValueError(
                        "fill-token targets length does not match decoder token count: "
                        f"{fill_targets.shape[0]} vs {outputs['fill_token_logits'].shape[0]}"
                    )
                fill_metrics = self._binary_head_metrics(outputs["fill_token_logits"], fill_targets)
                fill_token_loss = fill_metrics["loss"]
                fill_token_acc = fill_metrics["acc"]
                fill_token_target_pos_rate = fill_metrics["target_pos_rate"]
                fill_token_pred_pos_rate = fill_metrics["pred_pos_rate"]

        total = (
            float(self.loss_cfg.get("lambda_ce", 1.0)) * ce_recon
            + float(self.loss_cfg.get("lambda_aux_regression", 0.0)) * aux_regression
            + float(self.loss_cfg.get("lambda_direct_regression", 0.0)) * direct_regression
            + float(self.loss_cfg.get("lambda_subdiv", 0.0)) * subdiv_loss
            + float(self.loss_cfg.get("lambda_kl", 0.0)) * kl
            + float(self.loss_cfg.get("lambda_consistency", 0.0)) * consistency
            + self.lambda_topo_edge * topology_edge_loss
            + self.lambda_topo_open_boundary * topology_open_boundary_loss
            + self.lambda_structure_token * structure_token_loss
            + self.lambda_fill_token * fill_token_loss
        )

        acc_bins = (pred_bins == gt_bins).float().mean()
        face_match = (pred_bins == gt_bins).reshape(-1, 3, 3).all(dim=-1).all(dim=-1).float()
        acc_face = face_match.mean()
        acc_vertex_v0 = (pred_bins[:, 0:3] == gt_bins[:, 0:3]).all(dim=-1).float().mean()
        acc_vertex_v1 = (pred_bins[:, 3:6] == gt_bins[:, 3:6]).all(dim=-1).float().mean()
        acc_vertex_v2 = (pred_bins[:, 6:9] == gt_bins[:, 6:9]).all(dim=-1).float().mean()
        mae_bins = (pred_bins.float() - gt_bins.float()).abs().mean()
        offset_mae_soft = (pred_offsets_soft - gt_offsets).abs().mean()
        offset_mae_argmax = (pred_offsets_argmax - gt_offsets).abs().mean()

        gt_vertices = normalized_offsets_to_vertices(
            coords_xyz=batch["coords_xyz"].to(gt_offsets.device),
            normalized_offsets=gt_offsets,
            max_offset_per_face=batch["max_offset_per_face"].to(gt_offsets.device),
            resolution=self.resolution,
        )
        pred_vertices_argmax = normalized_offsets_to_vertices(
            coords_xyz=batch["coords_xyz"].to(gt_offsets.device),
            normalized_offsets=pred_offsets_argmax,
            max_offset_per_face=batch["max_offset_per_face"].to(gt_offsets.device),
            resolution=self.resolution,
        )
        vertex_mae_abs = (pred_vertices_argmax - gt_vertices).abs().mean()
        vertex_rmse_abs = torch.sqrt(((pred_vertices_argmax - gt_vertices) ** 2).mean())

        return {
            "loss": total,
            "ce_recon": ce_recon.detach(),
            "ce_v0": ce_v0.detach(),
            "ce_v1": ce_v1.detach(),
            "ce_v2": ce_v2.detach(),
            "aux_regression": aux_regression.detach(),
            "direct_regression": direct_regression.detach(),
            "subdiv_loss": subdiv_loss.detach(),
            "kl": kl.detach(),
            "consistency": consistency.detach(),
            "topology_edge_loss": topology_edge_loss.detach(),
            "topology_open_boundary_loss": topology_open_boundary_loss.detach(),
            "topology_edge_acc": topology_edge_acc.detach(),
            "topology_open_boundary_acc": topology_open_boundary_acc.detach(),
            "topology_edge_target_pos_rate": topology_edge_target_pos_rate.detach(),
            "topology_open_boundary_target_pos_rate": topology_open_boundary_target_pos_rate.detach(),
            "topology_edge_pred_pos_rate": topology_edge_pred_pos_rate.detach(),
            "topology_open_boundary_pred_pos_rate": topology_open_boundary_pred_pos_rate.detach(),
            "structure_token_loss": structure_token_loss.detach(),
            "fill_token_loss": fill_token_loss.detach(),
            "structure_token_acc": structure_token_acc.detach(),
            "fill_token_acc": fill_token_acc.detach(),
            "structure_token_target_pos_rate": structure_token_target_pos_rate.detach(),
            "fill_token_target_pos_rate": fill_token_target_pos_rate.detach(),
            "structure_token_pred_pos_rate": structure_token_pred_pos_rate.detach(),
            "fill_token_pred_pos_rate": fill_token_pred_pos_rate.detach(),
            "acc_bins": acc_bins.detach(),
            "acc_face": acc_face.detach(),
            "acc_vertex_v0": acc_vertex_v0.detach(),
            "acc_vertex_v1": acc_vertex_v1.detach(),
            "acc_vertex_v2": acc_vertex_v2.detach(),
            "mae_bins": mae_bins.detach(),
            "offset_mae_soft": offset_mae_soft.detach(),
            "offset_mae_argmax": offset_mae_argmax.detach(),
            "vertex_mae_abs": vertex_mae_abs.detach(),
            "vertex_rmse_abs": vertex_rmse_abs.detach(),
            "pred_bins": pred_bins.detach(),
        }

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        if batch is None:
            return None
        outputs = self._forward_impl(
            batch=batch,
            sample_posterior=self.train_behavior["sample_posterior"],
            teacher_forcing=self.train_behavior["teacher_forcing"],
            guided_structure=self.train_behavior["guided_structure"],
        )
        terms = self._compute_loss_terms(batch, outputs)
        self.log("train/loss", terms["loss"], on_step=True, on_epoch=True, prog_bar=True, batch_size=batch["batch_size"])
        self.log("train/total", terms["loss"], on_step=True, on_epoch=True, prog_bar=False, batch_size=batch["batch_size"])
        self.log("train/acc", terms["acc_bins"], on_step=True, on_epoch=True, prog_bar=False, batch_size=batch["batch_size"])
        for key in (
            "ce_recon",
            "ce_v0",
            "ce_v1",
            "ce_v2",
            "aux_regression",
            "direct_regression",
            "subdiv_loss",
            "kl",
            "consistency",
            "topology_edge_loss",
            "topology_open_boundary_loss",
            "topology_edge_acc",
            "topology_open_boundary_acc",
            "topology_edge_target_pos_rate",
            "topology_open_boundary_target_pos_rate",
            "topology_edge_pred_pos_rate",
            "topology_open_boundary_pred_pos_rate",
            "structure_token_loss",
            "fill_token_loss",
            "structure_token_acc",
            "fill_token_acc",
            "structure_token_target_pos_rate",
            "fill_token_target_pos_rate",
            "structure_token_pred_pos_rate",
            "fill_token_pred_pos_rate",
            "acc_bins",
            "acc_face",
            "acc_vertex_v0",
            "acc_vertex_v1",
            "acc_vertex_v2",
            "mae_bins",
            "offset_mae_soft",
            "offset_mae_argmax",
            "vertex_mae_abs",
            "vertex_rmse_abs",
        ):
            self.log(f"train/{key}", terms[key], on_step=True, on_epoch=True, prog_bar=key in {"ce_recon", "acc_bins", "acc_face"}, batch_size=batch["batch_size"])
        self.log("train/num_tokens", float(batch["coords"].shape[0]), on_step=True, on_epoch=False, prog_bar=False, batch_size=batch["batch_size"])
        self.log("train/adj_pairs", float(batch["adj_fi"].numel()), on_step=True, on_epoch=False, prog_bar=False, batch_size=batch["batch_size"])
        return terms["loss"]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        if batch is None:
            return None
        outputs = self._forward_impl(
            batch=batch,
            sample_posterior=self.val_behavior["sample_posterior"],
            teacher_forcing=self.val_behavior["teacher_forcing"],
            guided_structure=self.val_behavior["guided_structure"],
        )
        terms = self._compute_loss_terms(batch, outputs)
        self.log("val/loss", terms["loss"], on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["batch_size"], sync_dist=True)
        self.log("val/total", terms["loss"], on_step=False, on_epoch=True, batch_size=batch["batch_size"], sync_dist=True)
        self.log("val/acc", terms["acc_bins"], on_step=False, on_epoch=True, batch_size=batch["batch_size"], sync_dist=True)
        for key in (
            "ce_recon",
            "ce_v0",
            "ce_v1",
            "ce_v2",
            "aux_regression",
            "direct_regression",
            "subdiv_loss",
            "kl",
            "consistency",
            "topology_edge_loss",
            "topology_open_boundary_loss",
            "topology_edge_acc",
            "topology_open_boundary_acc",
            "topology_edge_target_pos_rate",
            "topology_open_boundary_target_pos_rate",
            "topology_edge_pred_pos_rate",
            "topology_open_boundary_pred_pos_rate",
            "structure_token_loss",
            "fill_token_loss",
            "structure_token_acc",
            "fill_token_acc",
            "structure_token_target_pos_rate",
            "fill_token_target_pos_rate",
            "structure_token_pred_pos_rate",
            "fill_token_pred_pos_rate",
            "acc_bins",
            "acc_face",
            "acc_vertex_v0",
            "acc_vertex_v1",
            "acc_vertex_v2",
            "mae_bins",
            "offset_mae_soft",
            "offset_mae_argmax",
            "vertex_mae_abs",
            "vertex_rmse_abs",
        ):
            self.log(f"val/{key}", terms[key], on_step=False, on_epoch=True, batch_size=batch["batch_size"], sync_dist=True)

    @torch.no_grad()
    def reconstruct_batch(self, batch: Dict[str, Any], free_run: bool = False) -> Dict[str, Any]:
        outputs = self._forward_impl(
            batch=batch,
            sample_posterior=False,
            teacher_forcing=False,
            guided_structure=not free_run,
        )
        pred_bins = self._compose_pred_bins(outputs)
        return {
            "pred_sparse": outputs["pred_sparse"],
            "pred_bins": pred_bins,
            "topology_logits_edge": outputs["topology_logits_edge"],
            "topology_logits_open_boundary": outputs["topology_logits_open_boundary"],
            "structure_token_logits": outputs["structure_token_logits"],
            "fill_token_logits": outputs["fill_token_logits"],
        }


AnchorFaceVaeModule = QuantizedFaceVaeModule

