artistic_mesh_vae
=================

Minimal Trellis2-style sparse SC-VAE experiments for artistic mesh autoencoding
with quantized face anchors.

This package is intentionally narrow:

- build mixed-source low-face candidate manifests
- convert meshes into sparse voxel face tokens
- train a small SC-VAE with Trellis2 sparse modules
- export simple reconstruction previews

Night-one scope is overfit-first. The initial token format is:

- `3D anchor_local` in voxel units relative to the voxel center
- `9D vertex_rel` as three vertex offsets relative to the face anchor

The first collision policy is deliberately strict: reject meshes whose anchor
quantization exceeds the confi