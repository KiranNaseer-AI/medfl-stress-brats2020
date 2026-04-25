# MedFL-Stress: Robustness Evaluation of Federated Brain Tumor Segmentation Under Cross-Hospital MRI Heterogeneity

> **FORGE-BENCH** | Part of the [FORGE Framework](https://github.com/KiranNaseer-AI) for Out-of-distribution Robustness, Generalisation and Evaluation

[![Paper](https://img.shields.io/badge/Paper-MIUA_2026-blue?style=flat)](https://miua2026.org)
[![Dataset](https://img.shields.io/badge/Dataset-BraTS_2020-green?style=flat)](https://www.med.upenn.edu/cbica/brats2020/data.html)
[![Framework](https://img.shields.io/badge/Framework-Flower_FL-orange?style=flat)](https://flower.ai)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python&logoColor=white)](https://python.org)

---

## What This Paper Is About

Federated learning (FL) promises to train medical AI across hospitals without sharing raw patient data. But standard FL benchmarks assume away the hardest real-world problem: MRI scanners at different hospitals produce images that look systematically different — different intensities, contrasts, noise profiles, and acquisition protocols — even for the same pathology.

**MedFL-Stress** is a stress-testing framework that asks: *how badly does cross-hospital MRI appearance heterogeneity hurt federated brain tumor segmentation, and which FL algorithms are most robust to it?*

We simulate realistic cross-site distribution shift by partitioning the BraTS 2020 dataset into heterogeneous client splits, each representing a hospital with a different MRI acquisition profile. We then evaluate standard FL algorithms — FedAvg, FedProx, and FedNova — under progressively severe appearance shift, measuring both average Dice score and **worst-client performance** — the metric that matters most when a hospital's patients are systematically underserved.

### Key Findings

- Average Dice scores mask severe per-client degradation under high heterogeneity
- Worst-client Dice drops by up to **X%** from α=1.0 to α=0.1 under FedAvg
- FedProx provides marginal robustness gains but does not resolve worst-client collapse
- Appearance heterogeneity is a distinct failure mode from label heterogeneity, requiring dedicated evaluation protocols

> This work contributes **FORGE-BENCH** — the benchmarking component of the FORGE evaluation framework — establishing standardised stress-test conditions for federated medical imaging research.

---

## Repository Structure

```
medfl-stress-brats2020/
├── data/
│   └── partition/          # Scripts to create heterogeneous client splits
├── models/
│   └── unet.py             # 3D U-Net segmentation model
├── federated/
│   ├── fedavg.py           # FedAvg implementation
│   ├── fedprox.py          # FedProx implementation
│   └── fednova.py          # FedNova implementation
├── evaluation/
│   ├── metrics.py          # Dice, worst-client Dice, Hausdorff distance
│   └── report.py           # Per-client result aggregation
├── configs/
│   └── default.yaml        # Experiment configuration
├── train_federated.py      # Main training entry point
├── evaluate.py             # Evaluation entry point
├── requirements.txt
└── README.md
```

> **Note:** Code upload is in progress. The repository will be fully populated upon paper acceptance at MIUA 2026. Watch this repo to be notified.

---

## Dataset

We use the **BraTS 2020** (Brain Tumor Segmentation) dataset.

| Property | Details |
|----------|---------|
| Modalities | T1, T1ce, T2, FLAIR |
| Task | Multi-class brain tumor segmentation (WT, TC, ET) |
| Subjects | 369 training cases with ground truth |
| Source | University of Pennsylvania (UPenn CBICA) |
| Access | Requires registration at Synapse (syn23193431) |

### Download Instructions

1. Register at [Synapse](https://www.synapse.org/#!Synapse:syn23193431) — free academic account
2. Request access to the BraTS 2020 training data
3. Download and place data at `data/raw/BraTS2020_TrainingData/`

### Creating Heterogeneous Client Splits

We simulate cross-hospital heterogeneity by applying intensity normalisation variations per client, controlled by a Dirichlet parameter α:

```bash
python data/partition/create_splits.py \
    --data_dir data/raw/BraTS2020_TrainingData \
    --num_clients 5 \
    --alpha 0.1 \
    --seed 42 \
    --output_dir data/splits/
```

Lower α means more severe appearance heterogeneity across simulated hospitals.

---

## Setup

### Requirements

- Python 3.8+
- CUDA 11.3+ (GPU required for 3D U-Net training)
- ~50 GB disk space for BraTS 2020

### Installation

```bash
git clone https://github.com/KiranNaseer-AI/medfl-stress-brats2020.git
cd medfl-stress-brats2020
pip install -r requirements.txt
```

Core dependencies:

```
torch>=1.12.0
flwr>=1.0.0
monai>=1.0.0
nibabel>=4.0.0
numpy>=1.23.0
pyyaml>=6.0
```

---

## Running Experiments

### 1. Federated training

```bash
python train_federated.py \
    --algorithm fedavg \
    --num_clients 5 \
    --num_rounds 50 \
    --alpha 0.1 \
    --data_dir data/splits/ \
    --output_dir results/fedavg_alpha01/
```

Swap `--algorithm` for `fedprox` or `fednova`. Vary `--alpha` across `{0.1, 0.3, 0.5, 1.0}` to reproduce the heterogeneity sweep.

### 2. Evaluation

```bash
python evaluate.py \
    --checkpoint results/fedavg_alpha01/best_model.pt \
    --data_dir data/splits/ \
    --output_dir results/fedavg_alpha01/eval/
```

This produces per-client Dice scores, worst-client Dice, and Hausdorff distance — the three metrics reported in the paper.

### 3. Reproducing the full stress-test sweep

```bash
bash scripts/run_all_experiments.sh
```

This runs all algorithm × α combinations and writes a consolidated CSV for plotting.

---

## Citation

If you use MedFL-Stress or the FORGE-BENCH evaluation protocol in your work, please cite:

```bibtex
@inproceedings{naseer2026medflstress,
  title     = {MedFL-Stress: Robustness Evaluation of Federated Brain Tumor
               Segmentation Under Cross-Hospital MRI Appearance Heterogeneity},
  author    = {Naseer, Kiran and Chaudhry, Nauman Riaz},
  booktitle = {Medical Image Understanding and Analysis (MIUA)},
  year      = {2026},
  note      = {Under Review}
}
```

---

## FORGE Framework

This repository is the **FORGE-BENCH** component of the FORGE research programme:

| Component | Role | Repo / Paper |
|-----------|------|--------------|
| **FORGE-BENCH** | Stress-test benchmarking | This repo |
| **FORGE-DIAG** | Diagnosing VLM instability | [multimodal-domain-shift-vlm](https://github.com/KiranNaseer-AI/multimodal-domain-shift-vlm) |
| **FORGE-EVAL** | Worst-client FL NLP evaluation | EMNLP 2026 (in progress) |
| **FORGE-ADAPT** | Adaptive robustness fine-tuning | Chapter 4 (in progress) |

---

## Author

**Kiran Naseer** — PhD Researcher, University of Gujrat, Pakistan  
Supervised by Dr. Naveed Anwer Butt

[![Google Scholar](https://img.shields.io/badge/Google_Scholar-4285F4?style=flat&logo=google-scholar&logoColor=white)](https://scholar.google.com/citations?user=Ek9e3qwAAAAJ&hl=en)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kiran-naseer-37925b329)
[![GitHub](https://img.shields.io/badge/GitHub-KiranNaseer--AI-black?style=flat&logo=github)](https://github.com/KiranNaseer-AI)
[![ORCID](https://img.shields.io/badge/ORCID-A6CE39?style=flat&logo=orcid&logoColor=white)](https://orcid.org/0009-0005-5129-8155)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
