# Text-quality GRPO training

This repository contains the standalone GRPO launcher and text-quality reward
plugin used for the GenProve training run. Every script discovers the repository
root from its own location, so the repository can be cloned to any directory.

## Deploy

Clone and enter the repository wherever you want it to live:

```bash
git clone git@github.com:yanghaoyuliao/Text_quality_rl.git Text_quality_rl
cd Text_quality_rl
```

By default, paths are relative to the repository root:

| Resource | Default location |
| --- | --- |
| Python environment | `.venv` |
| GRPO model | `models/merge_model/GenProve_sft_200` |
| Similarity model | `models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| GRPO dataset | `data/train_data/en_rl_annotation_with_refs.jsonl` |
| Training output | `models/rl_train_model` |

The dataset is deliberately not committed to GitHub. Copy the authorized JSONL
file to the default location, or set `GENPROVE_DATASET_PATH` to an alternate
absolute or relative path before validation/training.

## Install the environment

On an NVIDIA server with CUDA 12.4:

```bash
bash scripts/setup_grpo_env.sh
```

The dependency file pins the versions tested in the original environment and
installs `ms-swift` from the exact source revision used there.

## Download the models

Pass the ModelScope token through the environment (it is never stored in this
repository or in the command line):

```bash
read -r -s MODELSCOPE_API_TOKEN
export MODELSCOPE_API_TOKEN
bash scripts/download_models.sh
```

This downloads:

- `LeonYoung/GenProve_sft_200`
- `LeonYoung/paraphrase-multilingual-MiniLM-L12-v2`

## Validate and train

After placing the dataset at its default location (or exporting
`GENPROVE_DATASET_PATH`):

```bash
bash scripts/validate_grpo_inputs.sh
bash code/training/grpo_multiturn.sh
```

The launcher uses eight GPUs (`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`),
DeepSpeed ZeRO-3, LoRA, and the `text_quality_reward` plugin. Set
`SWANLAB_API_KEY` if the SwanLab account requires authenticated logging.

## Path overrides

All defaults can be overridden without editing scripts:

| Variable | Applies to |
| --- | --- |
| `GENPROVE_ENV_DIR` | Python environment and launchers |
| `GENPROVE_REQUIREMENTS_FILE` | Dependency file |
| `GENPROVE_MODEL_PATH` | GRPO model |
| `TEXT_QUALITY_RELEVANCE_MODEL` | Similarity model |
| `GENPROVE_DATASET_PATH` | Training dataset |
| `GENPROVE_OUTPUT_DIR` | Training output |
| `GENPROVE_REWARD_PLUGIN` | Reward plugin |

For example, to keep models and data outside the clone:

```bash
export GENPROVE_MODEL_PATH=/data/models/GenProve_sft_200
export TEXT_QUALITY_RELEVANCE_MODEL=/data/models/paraphrase-multilingual-MiniLM-L12-v2
export GENPROVE_DATASET_PATH=/data/train/en_rl_annotation_with_refs.jsonl
export GENPROVE_OUTPUT_DIR=/data/models/rl_train_model
bash scripts/validate_grpo_inputs.sh
bash code/training/grpo_multiturn.sh
```

## Files

- `code/training/grpo_multiturn.sh`: portable training launcher.
- `code/training/Text_quality_only_annotation.py`: sentence-level similarity
  and ROUGE-L reward.
- `requirements-training.txt`: pinned GPU training dependencies.
- `scripts/setup_grpo_env.sh`: creates the virtual environment and installs dependencies.
- `scripts/download_models.sh`: downloads both ModelScope models.
- `scripts/validate_grpo_inputs.sh`: checks paths and runs a similarity-model
  embedding smoke test.
