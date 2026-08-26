# Text-quality GRPO training

This repository contains the standalone GRPO launcher and text-quality reward
plugin used for the GenProve training run. The scripts intentionally use
absolute paths so the deployment is deterministic on the target server.

## Deployment layout

Clone this repository at exactly:

```bash
git clone git@github.com:yanghaoyuliao/Text_quality_rl.git /usr/data/wjx/genprove_2
```

The scripts expect these absolute paths:

| Resource | Path |
| --- | --- |
| Python environment | `/usr/data/wjx/genprove_2/.venv` |
| GRPO model | `/usr/data/wjx/genprove_2/models/merge_model/GenProve_sft_200` |
| Similarity model | `/usr/data/wjx/trove/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| GRPO dataset | `/usr/data/wjx/genprove_2/data/train_data/en_rl_annotation_with_refs.jsonl` |
| Training output | `/usr/data/wjx/genprove_2/models/rl_train_model` |

The two model directories are downloaded from ModelScope. The training dataset
is deliberately not committed to GitHub; place the authorized copy at the
absolute path above before starting training.

## Install the environment

On an NVIDIA server with CUDA 12.4:

```bash
bash /usr/data/wjx/genprove_2/scripts/setup_grpo_env.sh
```

The dependency file pins the versions tested in the original environment and
installs `ms-swift` from the exact source revision used there.

## Download the models

Read the access token without putting it in shell history, then run:

```bash
read -r -s MODELSCOPE_API_TOKEN
export MODELSCOPE_API_TOKEN
bash /usr/data/wjx/genprove_2/scripts/download_models.sh
```

This downloads:

- `LeonYoung/GenProve_sft_200`
- `LeonYoung/paraphrase-multilingual-MiniLM-L12-v2`

## Validate and train

After copying the dataset to the absolute path above:

```bash
bash /usr/data/wjx/genprove_2/scripts/validate_grpo_inputs.sh
bash /usr/data/wjx/genprove_2/code/training/grpo_multiturn.sh
```

The launcher uses eight GPUs (`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`),
DeepSpeed ZeRO-3, LoRA, and the `text_quality_reward` plugin. Set
`SWANLAB_API_KEY` if the SwanLab account requires authenticated logging.

## Files

- `code/training/grpo_multiturn.sh`: absolute-path training launcher.
- `code/training/Text_quality_only_annotation.py`: sentence-level similarity
  and ROUGE-L reward.
- `requirements-training.txt`: pinned GPU training dependencies.
- `scripts/setup_grpo_env.sh`: creates `/usr/data/wjx/genprove_2/.venv` and
  installs dependencies.
- `scripts/download_models.sh`: downloads both ModelScope models.
- `scripts/validate_grpo_inputs.sh`: checks paths and runs a similarity-model
  embedding smoke test.
