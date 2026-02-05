# When Grouped Cyclic Shift meets Masked Image Modeling: Effective Pre-training for Data-scarce 3D Ultrasound Analysis Tasks

This repo implements a 3D medical imaging pipeline inspired by:
- **HySparK**: Hybrid Sparse Masking for Large Scale Medical Image Pre-Training (MICCAI 2024).  
- **MAE**: Masked Autoencoders Are Scalable Vision Learners (He et al., 2021).

We support:
- **Pretraining** (MIM): sparse encoder + densify + lightweight decoder, reconstruct masked voxels
- **Finetuning** (segmentation): supervised 3D segmentation using pretrained weights (optional)
- **Testing/Inference**: sliding-window inference with MONAI

---

## 1. Directory Structure

```
GCSMIM/
|---- pretrain.py                     # pretraining entry
|---- finetune.py                     # finetune entry
|---- test.py                         # evaluation/testing entry
|
|---- engine/
|     |---- __init__.py               # (optional) make engine a package
|     |---- pretrain.py               # pretraining train loop
|     |---- finetune.py               # finetune train loop + evaluate()
|
|---- models/
|     |---- __init__.py               # build_sparse_encoder(...)
|     |---- encoder.py
|     |---- decoder.py
|     |---- gcsmim.py                 # pretrain wrapper (sparse encode + densify + decode + loss)
|     |
|     |---- network/
|           |---- __init__.py         # (optional) make network a package
|           |---- gcsmim_model.py     # segmentation network + build_gcsmim(...)
|
|---- utils/
|     |---- __init__.py               # (optional) make utils a package
|     |---- misc.py
|     |---- datasets.py
|     |---- lr_sched.py
|     |---- loss.py                   # HybridSegLoss + DiceLoss3D (kept)
|
|---- data_root/
|     |---- dataset.json
|     |
|     |---- imagesTr/
|     |     |---- case_0001.nii.gz
|     |     |---- case_0002.nii.gz
|     |
|     |---- labelsTr/
|     |     |---- case_0001.nii.gz
|     |     |---- case_0002.nii.gz
|     |
|     |---- imagesTs/
|     |     |---- case_1001.nii.gz
|     |
|     |---- labelsTs/
|           |---- case_1001.nii.gz
```
---

## 2. Environment

### Recommended
- Python **3.10** (recommended; avoids `int | None` annotation errors)
- PyTorch + MONAI + nibabel

### Install (example, CUDA 11.8)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install monai nibabel numpy tqdm timm tensorboard
```

## 3. Data Format

### 3.1 Dataset Root Layout
Your `data_dir` must contain a `dataset.json` and the corresponding files (NIfTI `.nii` / `.nii.gz` or other MONAI-supported formats).

### 3.2 dataset.json schema (required)
We use three splits: `pretrain`, `finetune`, `test`.

**Rule**: `test` should not overlap with `finetune`. `pretrain` can be unlabeled and may come from separate pools.

Example `dataset.json`:
```json
{
  "pretrain": [
    {"image": "imagesTr/case_0001.nii.gz"},
    {"image": "imagesTr/case_0002.nii.gz"}
  ],
  "finetune": [
    {"image": "imagesTr/case_0003.nii.gz", "label": "labelsTr/case_0003.nii.gz"},
    {"image": "imagesTr/case_0004.nii.gz", "label": "labelsTr/case_0004.nii.gz"}
  ],
  "test": [
    {"image": "imagesTs/case_1001.nii.gz", "label": "labelsTs/case_1001.nii.gz"}
  ]
}
```

> Paths can be relative (recommended) — they will be resolved relative to `data_dir`.


## 4. How to Run

> Always run commands from **project root** (the folder that contains `scripts/`, `models/`, `utils/`).

### 4.1 Pretrain (masked reconstruction)
```bash
python -m scripts.pretrain   --data_path  /path/to/data_root   --output_dir /path/to/output/pretrain   --input_size 96   --batch_size 1   --epochs 5   --mask_ratio 0.75   --device cuda
```

### 4.2 Finetune (segmentation)
```bash
python -m scripts.finetune   --data_path  /path/to/data_root   --output_dir /path/to/output/finetune   --input_size 96   --num_classes 3   --batch_size 1   --epochs 5   --device cuda   --loss_type hybrid   --pretrained_ckpt /path/to/output/pretrain/checkpoint-pretrain-best.pth
```

Notes:
- `--pretrained_ckpt` is optional. If provided, we load encoder weights from keys prefixed by `sparse_encoder.` and load into finetune model with `strict=False`.

### 4.3 Test / Inference (sliding window)
```bash
python -m scripts.test   --data_path /path/to/data_root   --ckpt /path/to/output/finetune/checkpoint-ft-best.pth   --num_classes 3   --roi_size 96 96 96   --sw_batch_size 2   --overlap 0.25   --device cuda   --output_dir /path/to/output/test
```

---

## 5. Common Errors & Fixes

### 5.1 PyTorch 2.6 `weights_only` / MONAI MetaTensor cache error
If `CacheNTransDataset` tries to `torch.load(cachefile)` and fails with safe-unpickling error:
- easiest: **delete cache folders** (`cache_pretrain/ cache_finetune_train/ ...`) and re-run
- or allowlist MONAI MetaTensor (only if you trust the cache source)

Recommended in dev: delete cache directory when transforms change.


## 6. Citation

If you use this repo, please cite:

```bibtex
@inproceedings{tang2024hyspark,
  title={Hyspark: Hybrid sparse masking for large scale medical image pre-training},
  author={Tang, Fenghe and Xu, Ronghao and Yao, Qingsong and Fu, Xueming and Quan, Quan and Zhu, Heqin and Liu, Zaiyi and Zhou, S. Kevin},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={330--340},
  year={2024},
  organization={Springer}
}

@Article{MaskedAutoencoders2021,
  author  = {Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Doll{'a}r and Ross Girshick},
  journal = {arXiv:2111.06377},
  title   = {Masked Autoencoders Are Scalable Vision Learners},
  year    = {2021}
}
```