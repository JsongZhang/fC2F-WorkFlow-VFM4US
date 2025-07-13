<h1>
  <img src="assets/probe.png" width="50" style="vertical-align: left; margin-right: 0px;">VFM4US: Clinician-Inspired Coarse-to-Fine Pretraining of Visual Foundation Model for Ultrasound Imaging
</h1>

Jiansong Zhang, Wenting Chen, Yongjian Chen, Shunlan Liu, Yanli Wang, Yongji Wu, Guisong Liu, Xiaoling Luo, Yaning Han, Guorong Lyu and  Linlin Shen

This repository contains the official implementation of **VFM4US**, our ultrasound-specific visual foundation model trained using a coarse-to-fine self-supervised pretraining paradigm. The model aims to capture both organ-level and scan-plane-level semantics and generalizes well to various downstream ultrasound tasks.

---

## 📊 Framework Overviews

![VFM4US Pipeline](./assets/vfm4us_pipeline.jpg)

The figure above illustrates the overall training and evaluation pipeline. The pretraining stage mimics clinicians’ diagnostic reasoning by integrating coarse anatomical understanding, followed by fine-grained scan-plane semantics.

---

## 🚀 Getting Started with Pretraining

### 1. Environment Setup
```bash
conda create -n vfm4us python=3.x
conda activate vfm4us
pip install -r requirements.txt
```

### 2. Start Pretraining
You can initiate pretraining using:
```bash
python ./Pretraining/US-SSL/adv_main.py
```
- The configuration file defines the encoder type, masking strategy, loss weights, and scheduler settings.
- Pretraining logs and checkpoints will be saved in `logs/`.

---

## 📚 Pretraining Dataset

We use a **large-scale ultrasound dataset of 510,000+ images** across 13 anatomical regions. The dataset is structured in a coarse-to-fine format:
- `US-510K/` contains:
  - `coarse/` — [🔗](https://www.radimagenet.com/) Body part-level coarse dataset
  - `fine/` —  [🔗](https://zenodo.org/records/8367932) Standard-plane & dignosis level fine dataset

Due to data licensing, only processed features and pretrained weights are shared in this repo. For more dataset access, please get in touch with the original papers' corresponding authors.

---

## 🎯 Downstream Tasks

We evaluate VFM4US on multiple downstream tasks to assess its transferability.

| Task            | Folder                | Description                                |
|-----------------|-----------------------|--------------------------------------------|
| Classification  | `./Downstream/Classification/`     | Breast(3c), Fetal Standard Plane(8c)           |
| Segmentation    | `./Downstream/Segmentation/`     | Breast Lesion, Thyroid Lesion, Nerve, Fetal Abdominal, Colon            |
| Image Generation | `./Downstream/Generation/`     | Thyroid, Breast, Liver, Kidney, Carotid Artery |


Run any task with:
-  Readme in the subfolder 


---

## 🧠 Pretrained Weights

The following pretrained weights are available for direct use:

| Model     | Weight File                        | Notes                           |
|-----------|------------------------------------|----------------------------------|
| VFM4US    | [🔗](https://drive.google.com/drive/folders/1owCttbnll0-ZjNrG45d3bejOHww0k4js?usp=sharing)`VFM4US`    | Contains coarse-grained, fine-grained and coarse-to-fine pre-training weights.          |
| MAE       | [🔗](https://drive.google.com/drive/folders/1qDbgX7eVSqREpCIRgaw1-RhMYa2Lz7D0?usp=sharing)`MAE`   | Contains coarse-grained, fine-grained and coarse-to-fine pre-training weights.      |
| DINO-v2   | [🔗](https://drive.google.com/drive/folders/1voAal8qmRj7gN2Js89hOxv2E6HZftr_p?usp=sharing)`DINO-v2`    | Contains coarse-grained, fine-grained and coarse-to-fine pre-training weights.   |

You can load the VFM4US weights with:

```python
checkpoint = torch.load(args.pretrained, map_location="cpu")
state_dict = checkpoint.get('state_dict', checkpoint) 
new_state_dict = {}
linear_keyword = 'fc'  
for k in list(state_dict.keys()):
    if k.startswith('module.base_encoder') and not k.startswith(f'module.base_encoder.{linear_keyword}'):
       new_key = k.replace('module.base_encoder.', '')  
       new_state_dict[new_key] = state_dict[k]
net.encoder.load_state_dict(new_state_dict, strict=False)
```

---

### 🧪 Evaluation of Downstream Task Fine-Tuning based on VFM4US

[USFM](https://www.sciencedirect.com/science/article/abs/pii/S1361841524001270) is a previously reported coarse-grained ultrasound foundation model trained on US-3M, a private dataset ~6× larger than ours.  Segmentation uses DeepLabv3+/U-Net decoder, classification uses a 3-layer linear classifier, and generation uses Cycle-GAN.

---

#### 📊 Segmentation (DSC %)

| Pre-training | Data Size | Foundation Model | Thyroid | Breast | Colon | Fetal Abdomen | Neck Nerve |
|--------------|------------|------------------|---------|--------|-------|----------------|-------------|
| US-C2F | 510K(390K+120K) | MAE (75%)         | 59.52   | 58.27  | 55.65 | 73.76          | 78.40       |
| US-C2F | 510K(390K+120K) | MAE (50%)         | 54.77   | 55.70  | 55.27 | 72.55          | 72.70       |
| US-C2F | 510K(390K+120K) | MAE (25%)         | 56.24   | 56.65  | 56.56 | 73.16          | 72.34       |
| US-C2F | 510K(390K+120K) | Dino-v2 (ViT-b)   | 60.31   | 58.16  | 53.16 | 75.38          | 78.53       |
| US-3M  | 3M              | USFM (ViT-b)      | 57.51   | 59.52  | 60.19 | 77.57          | 78.21       |
| US-C2F | 510K(390K+120K) | VFM4US (ResNet18) | **80.01** | **95.09** | **64.24** | **98.31** | **80.26** |

---

#### 🧠 Classification (Top-1 %)

| Pre-training | Data Size | Foundation Model | Breast | Fetal Brain |
|--------------|-----------|------------------|--------|--------------|
| US-C2F | 510K(390K+120K) | MAE (75%)         | 82.52  | 88.89        |
| US-C2F | 510K(390K+120K) | MAE (50%)         | 84.86  | 90.42        |
| US-C2F | 510K(390K+120K) | MAE (25%)         | 84.59  | 90.59        |
| US-C2F | 510K(390K+120K) | Dino-v2 (ViT-b)   | 75.10  | 56.41        |
| US-3M  | 3M              | USFM (ViT-b)      | 75.63  | 71.79        |
| US-C2F | 510K(390K+120K) | VFM4US (ResNet18) | **86.47** | **91.45**  |

---

#### 🎨 Generation (SSIM %)

| Pre-training | Data Size | Foundation Model | Thyroid | Carotid | Liver | Kidney | Breast |
|--------------|-----------|------------------|---------|---------|-------|--------|--------|
| US-C2F | 510K(390K+120K) | MAE (75%)         | 48.44   | 35.50   | 53.30 | 52.00  | 35.59  |
| US-C2F | 510K(390K+120K) | MAE (50%)         | 48.40   | 31.74   | 54.44 | 51.51  | 37.50  |
| US-C2F | 510K(390K+120K) | MAE (50%)         | 47.10   | 29.54   | 54.70 | 52.10  | 35.91  |
| US-C2F | 510K(390K+120K) | Dino-v2 (ViT-b)   | 49.23   | 28.61   | 50.05 | 56.70  | 35.60  |
| US-3M  | 3M              | USFM (ViT-b)      | 50.66   | 35.28   | 57.76 | 53.90  | 39.20  |
| US-C2F | 510K(390K+120K) | VFM4US (ResNet18) | **51.44** | **39.98** | **58.36** | **62.34** | **40.28** |

## 🧾 Citation
Available soon...