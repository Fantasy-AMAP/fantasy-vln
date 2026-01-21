# FantasyVLN
This project provides the online evaluation and distributed data parallel training code for **FantasyVLN**. The online evaluation is implemented based on the [LH-VLN](https://github.com/HCPLab-SYSU/LH-VLN) benchmark, and the training code is built upon [ms-swift](https://github.com/modelscope/ms-swift) and [qwen-vl](https://github.com/QwenLM/Qwen3-VL).

## Introduction

![Framework](assets/framework.jpg)

**FantasyVLN** is a unified multimodal Chain-of-Thought (CoT) reasoning framework that enables efficient and precise navigation based on natural language instructions and visual observations. **FantasyVLN** combines the benefits of textual, visual, and multimodal CoT reasoning by constructing a unified representation space across these reasoning modes. To enable efficient reasoning, we align these CoT reasoning modes with non-CoT reasoning during training, while using only non-CoT reasoning at test time. Notably, we perform visual CoT in the latent space of a [VAR](https://github.com/FoundationVision/VAR) model, where only low-scale latent representations are predicted. Compared to traditional pixel-level visual CoT methods, our approach significantly improves both training and inference efficiency.

## Online Evaluation
We modify the [LH-VLN](https://github.com/HCPLab-SYSU/LH-VLN) codebase to support VLMs and multi-GPU inference.

### Installation
You can use the following commands to install the required environment, or refer to the LH-VLN environment setup tutorial for more details.
```bash
conda create -n fantasyvln_eval python=3.9
conda activate fantasyvln_eval
conda install habitat-sim==0.3.1 headless -c conda-forge -c aihabitat
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 xformers
pip install -r lhvln/requirements.txt
```

### Preparing Data

**HM3D**

LH-VLN uses [HM3D](https://aihabitat.org/datasets/hm3d/) as the scene dataset. The required data splits can be downloaded by following the command below. Note that an application must be submitted to [Matterport](https://matterport.com/legal/matterport-end-user-license-agreement-academic-use-model-data) before using the dataset. For more details, please refer to [this link](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d).

```bash
python -m habitat_sim.utils.datasets_download --username <api-token-id> --password <api-token-secret> --uids hm3d_train_v0.2
python -m habitat_sim.utils.datasets_download --username <api-token-id> --password <api-token-secret> --uids hm3d_val_v0.2
```

**LH-VLN**

LH-VLN dataset is available in [Hugging Face](https://huggingface.co/datasets/Starry123/LHPR-VLN) and [ModelScope](https://modelscope.cn/datasets/starry123/LHPR-VLN). The zipped files included in the downloaded dataset are not required for online evaluation.


Your final directory structure should be like this:

```
fantasy-vln/
├── lhvln/
│   ├── data/
│   │   ├── hm3d/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── hm3d_annotated_basis.scene_dataset_config.json
│   │   ├── task/
│   │   │   ├── batch_1/
│   │   │   ├── ...
│   │   │   └── batch_8/
│   │   ├── step_task/
│   │   │   ├── batch_1/
│   │   │   ├── ...
│   │   │   └── batch_8/
│   │   └── episode_task/
│   │       ├── batch_1.json.gz
│   │       ├── ...
│   │       └── batch_8.json.gz
```

## Run Evaluation

```bash
./eval.sh
```
You must specify the following parameters before runing the script:
- `HAB_GPU_ID`: GPU id used by Habitat-Sim for environment simulation; should be a valid physical GPU and not overlap with `RUN_GPU_IDS`.
- `RUN_GPU_IDS`: Comma-separated list of GPU ids for inference processes; each GPU launches one process and corresponds to a subset of test data.
- `SAVE_PATHS`: Comma-separated list of output directories where logs and evaluation results are saved.
- `MODEL_IDS`: Comma-separated list of model checkpoint paths; must have the same length and order as `SAVE_PATHS`.

## Training

### Installation
```bash
conda create -n fantasyvln_train python=3.10
conda activate fantasyvln_train
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 xformers
pip install requirements.txt
```

### Prepare Training Data
You can generate training data by runing the following commands:
```bash
hf download Starry123/LHPR-VLN batch_{1..8}.zip --repo-type dataset --local-dir ./data/images
for z in data/image/batch_*.zip; do unzip -o "$z" -d "${z%.zip}"; done

# Prepare non-CoT json data
python data/prepare_swift_data.py --set_name train --base_dir ./data/images --data_augmentation
python data/prepare_swift_data.py --set_name val --base_dir ./data/images --data_augmentation

# Prepare T-CoT json data
python data/prepare_tocot_data.py --excel_path data/tcot_annotations/excel_files --input_jsonl data/json_files/swift_his_20_train_aug.jsonl

# Prepare V-CoT json data
python data/prepare_tocot_data.py --scale_schedule 3 input_jsonl data/json_files/swift_his_20_train_aug.jsonl

# Prepare MM-CoT json data
python data/prepare_mmcot_data.py --vcot_json_path data/json_files/vcot_swift_his_20_train_aug.jsonl --tcot_json_path data/json_files/tcot_swift_his_20_train_aug.jsonl --save_as_ummcot_format True
```
PS: We used Qwen-VL-Max to generate textual CoT annotations for the data in `swift_his_20_train_aug.jsonl`. However, due to data licensing and privacy compliance considerations, we cannot release these annotations publicly. You may reproduce them by following the same procedure (describled in our paper).

The final directory structure should be like this:
```bash
fantasy-vln/
├── data/
│   ├── json_files/
│   │   ├── swift_his_20_train_aug.jsonl
│   │   ├── tcot_swift_his_20_train_aug.jsonl
│   │   ├── vcot_swift_his_20_train_aug.jsonl
│   │   ├── ummcot_swift_his_20_train_aug.jsonl
│   ├── images/
│   │   ├── batch_1
│   │   ├── batch_2
│   │   ├── batch_3
│   │   ├── batch_4
│   │   ├── batch_5
│   │   ├── batch_6
│   │   ├── batch_7
│   │   ├── batch_8
```

### Run Training
```bash
./train.sh
```
