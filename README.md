# mesh-vae

Standalone snapshot of the mesh VAE code used for the current `baseline`, `idea1`, `idea2`, and `idea4` training lines.

Contents:
- `artistic_mesh_vae/`: training entrypoint, model, data preprocessing, evaluation helpers, and launch scripts.
- `sandboxes/20260317_mesh_vae_armesh_switch/`: current `baseline`, `idea1`, and `idea4` configs, scripts, splits, and manifest.
- `sandboxes/20260318_mesh_vae_idea2_sparseified/`: current `idea2` sparse-dense build helper and cached split files.

Notes:
- The code depends on an external `TRELLIS.2` checkout on the training machines and is not vendored here.
- Runtime artifacts such as logs, checkpoints, caches, and experiment outputs are intentionally excluded.
