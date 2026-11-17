# Reward-Driven Point Selection for One-Shot Segmentation

This code trains and evaluates a policy that chooses five positive point prompts for a frozen Segment Anything Model (SAM). We compare behavioral cloning (BC), which imitates a greedy selector, with proximal policy optimization (PPO), which rewards improvements to the predicted mask.

We trained 13 selector variants with five seeds each using SAM ViT-B. For each run, we selected a checkpoint on 357 FSS-1000 validation episodes, then evaluated it with both ViT-B and ViT-H on two FSS-1000 test sets and on Kvasir-SEG (390 evaluations in total).

On the 450-episode FSS-1000 test set with ViT-H, BC with an auxiliary intersection-over-union (IoU) objective scores 0.599 mean IoU. PPO refinement raises this to 0.787 with mask feedback disabled in both models. Our reference policy (R5), which uses mask feedback and retains BC as a reference during training, scores 0.867 ± 0.012 (mean ± standard deviation across five seeds). The same policy scores 0.434 on Kvasir-SEG without adaptation. The [Zenodo results deposit](https://doi.org/10.5281/zenodo.24322568) includes the remaining comparisons and scores after each click.

## Install

The project supports Python 3.10–3.12 and uses `uv` to install the locked environment. The research extra installs [Segment Anything](https://github.com/facebookresearch/segment-anything/tree/dca509fe793f601edb92606367a655c15ac00fdf) at a fixed upstream revision. DINOv2 is loaded from its [fixed upstream revision](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8). Upstream source is fetched during installation or first model load; no upstream source is vendored in this repository.

```bash
git clone https://github.com/IMSEL-Lab/reward-driven-point-selection.git
cd reward-driven-point-selection
uv sync --locked --extra research
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

Dataset images and masks are not redistributed. Protocol manifests contain sample identities and relative paths, without image or mask files. They are available in the [Zenodo results deposit](https://doi.org/10.5281/zenodo.24322568). Install the downloaded protocol bundle with its verified SHA-256 digest:

```bash
uv run --no-sync python -m sam_rl.cli.install_protocol \
  --archive protocol-v2.zip \
  --sha256 aedca39b687f8fb63f55bd15450ac35826f328b9e71e2d56e96c0625341125c0
```

## Evaluate a released selector

To evaluate R5 (seed 17), download `results.zip` from the results deposit to the repository root. This example uses FSS-1000 and a CUDA GPU; Kvasir-SEG is not needed. After installing the protocol and weights above, extract the selector and run:

```bash
mkdir -p artifacts/released-checkpoints
unzip -jo results.zip results/checkpoints/R5-s17.pt -d artifacts/released-checkpoints
uv run --no-sync python -m sam_rl.cli.train_rl_finetune \
  --config configs/evaluation/r5_fss_test450_sam_b.yaml \
  --mode eval \
  --checkpoint artifacts/released-checkpoints/R5-s17.pt \
  --seed 17 \
  --split test \
  --device cuda:0
```

For a quick check on a computer without a GPU, replace `--device cuda:0` with `--device cpu --max-episodes 1`. The first fixed episode uses `moist_proof_pad/1` as support and `moist_proof_pad/10` as query (a `.jpg` image and a `.png` mask for each). With R5, seed 17, and ViT-B, the archived IoU after five clicks is **0.927999** for this episode. The CSV appears under `artifacts/evaluation/released_R5/`. Omit `--max-episodes` to evaluate all 450 episodes. To change the dataset root or decoder, edit a local copy of the configuration.

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

The [results archive, DOI 10.5281/zenodo.24322568](https://doi.org/10.5281/zenodo.24322568), contains the fixed episode lists, selected checkpoints, and evaluation scores for the five-seed study. Version 1.1.0 also includes the 660 PerSAM-F episode scores for Table 2 (66 FSS-1000 classes, mean IoU 0.7844). To check the archived scores without a GPU, datasets, or model downloads, extract `results.zip` and run `python3 results/check_deposit.py` from the directory where you extracted it.

The first-party code is licensed under [0BSD](LICENSE). Results, protocol manifests, and learned selector weights are dedicated to the public domain under CC0 1.0; the Zenodo helper script uses 0BSD. Third-party dependencies and datasets retain their own terms. Citation metadata is provided in `CITATION.cff`; citing the work is appreciated and is not a license requirement. See [the reproduction guide](docs/reproduction.md) for conditions, data requirements, commands, and scope limits.
