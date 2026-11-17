# Reproducing the controlled study

This guide describes the experiment configurations and execution sequence. The full campaign is compute intensive. First run and inspect the bounded pilot in the repository README.

## Study design

The study trains 13 point selectors with seeds 17, 29, 43, 71, and 101. Conditions B0–B3 are behavioral cloning controls; R0–R8 are PPO refinements that vary mask observation, initialization, and the BC policy used as a KL reference. Each policy selects five positive points on a 37 × 37 grid. SAM and DINOv2 stay frozen.

Training uses SAM ViT-B. The same selected checkpoint is evaluated with SAM ViT-B and ViT-H on FSS test450, FSS canonical660, and zero-shot Kvasir test450, for six evaluations per trained condition and seed. Checkpoint selection uses 357 clean FSS validation episodes. FSS test450 and canonical660 overlap and must be reported separately. Kvasir has 150 query images with three support examples per query; this is a single-dataset transfer evaluation.

The fixed episode manifests are distributed in the separate protocol archive; the archive contains metadata and relative paths but no images or masks. Download `protocol-v2.zip` from the separate results deposit and verify/install it with the release digest:

```bash
uv run --no-sync python -m sam_rl.cli.install_protocol \
  --archive protocol-v2.zip \
  --sha256 aedca39b687f8fb63f55bd15450ac35826f328b9e71e2d56e96c0625341125c0
```

Symlink your local FSS-1000 and Kvasir-SEG dataset directories to `data/FSS-1000/` and `data/Kvasir-SEG/`, the roots used by the released configs. Keep generated configs immutable after registry creation. If a different root is necessary, update a local template/specification before generation and write configs and registries to new output paths. Do not add datasets or generated outputs to a source commit.

## Dependencies and pretrained weights

Install the locked research environment and download the checkpoints using the README commands. The checkpoint downloader verifies SHA-256 before accepting a file. Model checkpoints are excluded from Git and need substantial local storage. The [SAM implementation](https://github.com/facebookresearch/segment-anything/tree/dca509fe793f601edb92606367a655c15ac00fdf) is installed as a dependency from its immutable upstream revision; [DINOv2](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8) code is loaded from its pinned upstream revision. These external components and their licenses remain subject to their upstream terms.

Use one device and local storage for the pilot. For a full study, verify available GPU memory, system RAM, dataset space, model-weight space, and scratch/output space before allocating jobs. Cache and run outputs are regenerated locally under `artifacts/`.

## Evaluate a released checkpoint

The [results deposit](https://doi.org/10.5281/zenodo.24322568) includes one selected policy checkpoint per condition and seed. Download `results.zip` to the repository root. With the protocol installed, FSS-1000 in `data/FSS-1000/`, and the foundation-model weights downloaded, run this example on a CUDA GPU:

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

This evaluates with SAM ViT-B. For a ViT-H evaluation, make a local copy of `configs/evaluation/r5_fss_test450_sam_b.yaml` and set its `models.eval_sam_checkpoint` and `models.eval_sam_model_type` fields to the ViT-H weight path and `vit_h`. Keep the tracked template unchanged. Evaluation requires the dataset and SAM/DINOv2 weights; the checkpoint archive contains selector weights only.

## Complete campaign

The default experiment specification is `configs/experiments/chair_v2.json`. Generate the full registry and per-condition configs:

```bash
uv run --no-sync python -m sam_rl.cli.experiment_suite generate \
  --spec configs/experiments/chair_v2.json \
  --registry artifacts/experiments/chair_v2/registry.json \
  --config-dir configs/generated/chair_v2/full
```

Validate the 65 registered training jobs, inspect their BC dependencies, and generate a dry-run plan before starting workers:

```bash
uv run --no-sync python -m sam_rl.cli.experiment_suite validate \
  --registry artifacts/experiments/chair_v2/registry.json
uv run --no-sync python -m sam_rl.cli.experiment_suite deps \
  --registry artifacts/experiments/chair_v2/registry.json
uv run --no-sync python -m sam_rl.cli.experiment_suite dryrun \
  --registry artifacts/experiments/chair_v2/registry.json \
  --gpus 0
```

The campaign runner accepts physical GPU IDs and optional job IDs. Partition the registry so every training job runs once, and do not schedule a job twice concurrently. The runner respects the BC prerequisites and writes per-job state and logs. A multi-GPU host can use several disjoint worker processes, each with its own GPU and non-overlapping job list. Confirm current GPU and disk availability yourself; the software does not reserve remote machines.

After all selected training jobs complete, generate evaluation plans from the frozen specification and registry:

```bash
uv run --no-sync python -m sam_rl.cli.evaluation_suite \
  --spec configs/experiments/chair_v2.json \
  --registry artifacts/experiments/chair_v2/registry.json \
  --seeds 17 29 43 71 101 \
  --output-root artifacts/experiments/chair_v2/evaluation \
  --config-root configs/generated/chair_v2/evaluation
```

The command above generates `evaluation_plan.json` and `collection.json` under the evaluation output root; it does not launch evaluations. After every selected checkpoint is present, execute pending plan jobs with `--run-plan`, for example: `uv run --no-sync python -m sam_rl.cli.evaluation_suite --spec configs/experiments/chair_v2.json --registry artifacts/experiments/chair_v2/registry.json --run-plan artifacts/experiments/chair_v2/evaluation/evaluation_plan.json --device cuda:0`. Evaluation needs the corresponding checkpoints and the same fixed episode manifests used during training and selection. It records source, checkpoint, configuration, and manifest provenance alongside results.

After plan execution, analyze a single complete evaluation collection directly. The collection must contain all expected condition, seed, benchmark, and decoder results for the analyzer's declared matrix.

```bash
uv run --no-sync python -m sam_rl.cli.analyze_experiments \
  --manifest artifacts/experiments/chair_v2/evaluation/collection.json \
  --output-dir artifacts/experiments/chair_v2/analysis
```

For multiple workers, first stage each collection and its referenced files without changing their contents. The merge specification has one entry per worker. `source_root` identifies that worker's checkout root, and `staged_artifact_root` identifies where the checkout's referenced paths have been copied beneath the merge workspace.

```json
{
  "sources": [
    {
      "collection": "input/worker-a/collection.json",
      "source_root": "input/worker-a/source",
      "staged_artifact_root": "staged/worker-a"
    },
    {
      "collection": "input/worker-b/collection.json",
      "source_root": "input/worker-b/source",
      "staged_artifact_root": "staged/worker-b"
    },
    {
      "collection": "input/worker-c/collection.json",
      "source_root": "input/worker-c/source",
      "staged_artifact_root": "staged/worker-c"
    }
  ]
}
```

With the specification saved to `artifacts/experiments/chair_v2/merge/merge_spec.json`, stage each worker collection and its source checkout while preserving repository-relative paths:

```text
artifacts/experiments/chair_v2/merge/
  merge_spec.json
  input/worker-a/collection.json
  input/worker-a/source/<repository-relative-path>
  input/worker-b/collection.json
  input/worker-b/source/<repository-relative-path>
  input/worker-c/collection.json
  input/worker-c/source/<repository-relative-path>
  staged/worker-a/<repository-relative-path>
  staged/worker-b/<repository-relative-path>
  staged/worker-c/<repository-relative-path>
```

Use the strict merger, then run the analyzer on the resulting complete collection:

```bash
uv run --no-sync python -m sam_rl.cli.merge_evaluation_collections \
  --merge-spec artifacts/experiments/chair_v2/merge/merge_spec.json \
  --merged-collection artifacts/experiments/chair_v2/merge/collection.json
uv run --no-sync python -m sam_rl.cli.analyze_experiments \
  --manifest artifacts/experiments/chair_v2/merge/collection.json \
  --output-dir artifacts/experiments/chair_v2/analysis
```

The worker collections must use the same evaluation source and non-overlapping seed subsets that cover all five seeds. Keep original per-episode files and failure records. The merger validates completion, expected run identities, and exact episode coverage before writing the merged collection.

## Limitations and scope

The reported results were produced with one frozen ViT-B training setup and five seeds. Evaluating those checkpoints with ViT-H does not measure training with ViT-H. The Kvasir evaluation tests one medical dataset and does not establish broad medical-domain generalization. The results archive includes PerSAM-F on the fixed 660-episode comparison set. It does not include negative prompts, DAgger, AlignSAM, or Plug-and-Play PPO. The full five-seed analysis reports training-seed standard deviations separately from episode-cluster bootstrap intervals.
