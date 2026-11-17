# Reward-Driven Point Selection for One-Shot Segmentation

This repository contains first-party training, evaluation, and analysis code for learning point prompts for a frozen Segment Anything Model (SAM). The project compares behavioral cloning (BC) with proximal policy optimization (PPO) under a five-positive-point budget. Images, masks, pretrained weights, checkpoints, and generated results are not stored here.

The controlled study has 13 selector conditions and five training seeds (65 training runs). Each selected checkpoint is evaluated on two FSS-1000 test sets and one Kvasir-SEG set with SAM ViT-B and ViT-H (390 evaluations). Training uses SAM ViT-B. Checkpoints selected on a clean 357-episode FSS validation set are evaluated with both decoders. Protocol metadata is distributed separately from code and data.

On the main FSS test set with SAM ViT-H, auxiliary-IoU BC scores 0.599 mean IoU; matched PPO refinement without mask feedback scores 0.787. The predeclared original-like PPO condition scores 0.867 ± 0.012 across five training seeds. The closed-loop mask-aware greedy heuristic scores 0.561 on this FSS test with SAM ViT-H, while similarity-weighted farthest-point sampling scores 0.774. Zero-shot performance on Kvasir-SEG is lower at 0.434. See the [Zenodo results deposit](https://doi.org/10.5281/zenodo.24226833) for full ablations, uncertainty estimates, per-click scores, and comparator details.

## Install

The project supports Python 3.10–3.12 and uses `uv` to install the locked environment. The research extra installs [Segment Anything](https://github.com/facebookresearch/segment-anything/tree/dca509fe793f601edb92606367a655c15ac00fdf) at a fixed upstream revision. DINOv2 is loaded from its [fixed upstream revision](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8). Upstream source is fetched during installation or first model load; no upstream source is vendored in this repository.

```bash
uv sync --extra research
```

Download the verified SAM and DINOv2 checkpoints into the ignored `artifacts/weights/` directory:

```bash
bash download_weights.sh
```

Obtain FSS-1000 and Kvasir-SEG separately. Symlink existing dataset directories into the expected local roots, so the images stay outside this repository:

```bash
mkdir -p data
ln -s /path/to/FSS-1000 data/FSS-1000
ln -s /path/to/Kvasir-SEG data/Kvasir-SEG
```

Dataset images and masks are not redistributed. Protocol manifests contain sample identities and relative paths, without image or mask files. They are available in the [Zenodo results deposit](https://doi.org/10.5281/zenodo.24226833). Install the downloaded protocol bundle with its verified SHA-256 digest:

```bash
uv run --no-sync python -m sam_rl.cli.install_protocol \
  --archive protocol-v2.zip \
  --sha256 aedca39b687f8fb63f55bd15450ac35826f328b9e71e2d56e96c0625341125c0
```

## Evaluate a released selector

The results deposit includes the 65 selected policy checkpoints in `results.zip`. To run R5 with seed 17 on the FSS test set, extract `results/checkpoints/R5-s17.pt` from `results.zip` into `artifacts/released-checkpoints/`, install the protocol archive, download the model weights, and place the dataset under `data/FSS-1000/`. Then run:

```bash
uv run --no-sync python -m sam_rl.cli.train_rl_finetune \
  --config configs/evaluation/r5_fss_test450_sam_b.yaml \
  --mode eval \
  --checkpoint artifacts/released-checkpoints/R5-s17.pt \
  --seed 17 \
  --split test \
  --device cuda:0
```

This evaluation config uses the ViT-B decoder. Evaluation writes outputs under the ignored `artifacts/` directory. For another dataset root or decoder, make a local copy of the config and edit it before evaluation.

## Run a pilot

Generate the five-condition pilot registry. It uses a small episode and update budget to check the environment before a complete campaign:

```bash
uv run --no-sync python -m sam_rl.cli.experiment_suite pilot \
  --registry artifacts/experiments/chair_v2/registry.json \
  --config-dir configs/generated/chair_v2/pilot \
  --pilot-train-episodes 32 \
  --pilot-val-episodes 6 \
  --pilot-steps 800
```

Validate the generated jobs, inspect their dependencies, then dry-run them before dispatch:

```bash
uv run --no-sync python -m sam_rl.cli.experiment_suite validate \
  --registry artifacts/experiments/chair_v2/pilot_registry.json
uv run --no-sync python -m sam_rl.cli.experiment_suite deps \
  --registry artifacts/experiments/chair_v2/pilot_registry.json
uv run --no-sync python -m sam_rl.cli.experiment_suite dryrun \
  --registry artifacts/experiments/chair_v2/pilot_registry.json \
  --gpus 0
```

After confirming the selected GPU is available, start the pilot with `run` and the same registry and GPU argument. The registry records each job state and checkpoint. For the complete five-seed study, follow [the reproduction guide](docs/reproduction.md). Do not run a full campaign without the required datasets, model weights, storage, and GPU capacity.

## Results and citation

Version 1.0.0 is available as a [GitHub release](https://github.com/IMSEL-Lab/reward-driven-point-selection/releases/tag/v1.0.0) and preserved in the [Zenodo code archive, DOI 10.5281/zenodo.24295124](https://doi.org/10.5281/zenodo.24295124).

Evaluation results and supporting files, including protocol metadata, selected policy checkpoints, and compact analysis tools, are available at [Zenodo, DOI 10.5281/zenodo.24226833](https://doi.org/10.5281/zenodo.24226833). This repository contains neither result archives, protocol metadata, datasets, nor trained checkpoints.

The first-party code is licensed under [0BSD](LICENSE). Results, protocol manifests, and learned selector weights are dedicated to the public domain under CC0 1.0; the Zenodo helper script uses 0BSD. Third-party dependencies and datasets retain their own terms. Citation metadata is provided in `CITATION.cff`; citing the work is appreciated and is not a license requirement. See [the reproduction guide](docs/reproduction.md) for conditions, data requirements, commands, and scope limits.
