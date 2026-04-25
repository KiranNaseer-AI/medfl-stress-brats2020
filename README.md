# MedFL-Stress: Robustness Evaluation of Federated Brain Tumor Segmentation Under Cross-Hospital MRI Heterogeneity

> **FORGE-BENCH** | Part of the [FORGE Framework](https://github.com/KiranNaseer-AI) for Out-of-distribution Robustness, Generalisation and Evaluation

[![Paper](https://img.shields.io/badge/Paper-MIUA_2026-blue?style=flat)](https://miua2026.org)
[![Dataset](https://img.shields.io/badge/Dataset-BraTS_2020-green?style=flat)](https://www.med.upenn.edu/cbica/brats2020/data.html)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)

---

## What This Paper Is About

Federated learning (FL) lets hospitals train segmentation models together without sharing patient data. But standard FL evaluation reports only the **average Dice score** across all hospitals — and that number hides a serious clinical problem.

A model that achieves a strong global mean may still fail consistently at one specific hospital. In clinical deployment, that is not a rounding error. It is a reliability failure that average reporting makes invisible.

**MedFL-Stress** is a controlled stress-testing framework designed to expose exactly that failure. We distribute 2D axial slices from the BraTS 2020 dataset across four simulated hospital clients and apply graded MRI appearance shifts — gamma contrast, scale-shift, and noise-plus-blur transformations — that reflect real scanner and acquisition variability in multi-site deployments.

We evaluate three widely used FL baselines: **FedAvg**, **FedProx**, and **FedBN**, treating **worst-hospital Dice** and **inter-hospital disparity** as primary evaluation targets, not supplementary observations.

### Key Results

| Algorithm | Mean Dice | Worst-Hospital Dice | Inter-Hospital Gap |
|-----------|-----------|--------------------|--------------------|
| FedAvg    | **0.8159** | 0.7309            | 0.0850            |
| FedProx   | 0.8085     | 0.7421            | 0.0664            |
| FedBN     | 0.8109     | **0.7656**        | **0.0503**        |

**The core finding:** FedAvg achieves the highest global mean Dice (0.8159), but beneath that number is an 8.5-point gap between its best and worst-performing hospital. FedBN — by retaining client-specific batch-normalisation statistics rather than folding them into the global aggregate — closes that gap by **41%** (0.0850 → 0.0503) while sacrificing less than half a Dice point in mean accuracy. The weakest hospital gains **3.5 Dice points outright** (0.7309 → 0.7656).

In a clinical context, those numbers are not minor differences that can be ignored.

> This work contributes **FORGE-BENCH** — the benchmarking component of the FORGE evaluation framework — establishing standardised worst-client stress-test protocols for federated medical imaging research.

---

## Dataset

We use the **BraTS 2020** (Brain Tumor Segmentation) dataset.

| Property | Details |
|----------|---------|
| Input | 2D axial slices from T1, T1ce, T2, FLAIR volumes |
| Task | Brain tumor segmentation |
| Clients | 4 simulated hospital clients |
| Shift types | Gamma contrast · Scale-shift · Noise + blur |
| Source | University of Pennsylvania CBICA |
| Access | Free registration at Synapse (syn23193431) |

### Download Instructions

1. Register at [Synapse](https://www.synapse.org/#!Synapse:syn23193431) — free academic account required
2. Request access to BraTS 2020 Training Data
3. Download and place at `data/raw/BraTS2020_TrainingData/`

### Creating the Four-Client Heterogeneous Split

```bash
python data/partition/create_splits.py \
    --data_dir data/raw/BraTS2020_TrainingData \
    --num_clients 4 \
    --shift_types gamma scale_shift noise_blur \
    --output_dir data/splits/
```

Each client receives a different appearance transformation severity, simulating scanner variability across hospital sites.

---

## Repository Structure

```
medfl-stress-brats2020/
├── data/
│   └── partition/          # Client split generation with appearance shifts
├── models/
│   └── unet2d.py           # 2D U-Net segmentation model
├── federated/
│   ├── fedavg.py           # FedAvg
│   ├── fedprox.py          # FedProx
│   └── fedbn.py            # FedBN (client-specific BN statistics)
├── evaluation/
│   ├── metrics.py          # Mean Dice, worst-hospital Dice, inter-hospital gap
│   └── report.py           # Per-client result aggregation and tables
├── configs/
│   └── default.yaml        # Experiment configuration
├── train_federated.py      # Main training entry point
├── evaluate.py             # Evaluation and reporting entry point
├── requirements.txt
└── README.md
```

> **Note:** Code is being prepared for release following paper acceptance at MIUA 2026. Watch this repo for updates.

---

## Setup

### Requirements

- Python 3.8+
- CUDA 11.3+ recommended
- ~30 GB disk space for BraTS 2020

### Installation

```bash
git clone https://github.com/KiranNaseer-AI/medfl-stress-brats2020.git
cd medfl-stress-brats2020
pip install -r requirements.txt
```

Core dependencies:

```
torch>=1.12.0
monai>=1.0.0
nibabel>=4.0.0
numpy>=1.23.0
scikit-learn>=1.0.0
pyyaml>=6.0
matplotlib>=3.5.0
```

---

## Running Experiments

### Training

```bash
python train_federated.py \
    --algorithm fedbn \
    --num_clients 4 \
    --num_rounds 50 \
    --data_dir data/splits/ \
    --output_dir results/fedbn/
```

Replace `--algorithm` with `fedavg` or `fedprox` to run the other baselines.

### Evaluation

```bash
python evaluate.py \
    --checkpoint results/fedbn/best_model.pt \
    --data_dir data/splits/ \
    --output_dir results/fedbn/eval/
```

This produces per-client Dice scores, worst-hospital Dice, and inter-hospital disparity — the three primary metrics from the paper.

### Full Stress-Test Sweep

```bash
bash scripts/run_all_experiments.sh
```

Runs all three algorithms across all appearance shift conditions and writes a consolidated results CSV.

---

## Citation

If you use MedFL-Stress or the FORGE-BENCH evaluation protocol, please cite:

```bibtex
@inproceedings{naseer2026medflstress,
  title     = {MedFL-Stress: Robustness Evaluation of Federated Brain Tumor
               Segmentation Under Cross-Hospital MRI Appearance Heterogeneity},
  author    = {Naseer, Kiran and Anwer, Naveed Anwer},
  booktitle = {Medical Image Understanding and Analysis (MIUA)},
  year      = {2026},
  note      = {Under Review}
}
```

---

## FORGE Framework

This repository is the **FORGE-BENCH** component of the FORGE research programme — a unified framework for out-of-distribution robustness, generalisation, and evaluation in federated and multimodal AI systems.

| Component | Role | Status |
|-----------|------|--------|
| **FORGE-BENCH** | Stress-test benchmarking — federated medical imaging | This repo · MIUA 2026 |
| **FORGE-DIAG** | Diagnosing VLM instability under physical domain shift | [multimodal-domain-shift-vlm](https://github.com/KiranNaseer-AI/multimodal-domain-shift-vlm) · ECCV 2026 |
| **FORGE-EVAL** | Worst-client robustness in federated NLP | EMNLP 2026 · In progress |
| **FORGE-ADAPT** | Adaptive gradient-signal stage switching | Chapter 4 · In progress |

---

## Author

**Kiran Naseer** — PhD Researcher, University of Gujrat, Pakistan
Supervised by Dr.Naveed Anwer Butt

[![Google Scholar](https://img.shields.io/badge/Google_Scholar-4285F4?style=flat&logo=google-scholar&logoColor=white)](https://scholar.google.com/citations?user=Ek9e3qwAAAAJ&hl=en)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kiran-naseer-37925b329)
[![GitHub](https://img.shields.io/badge/GitHub-KiranNaseer--AI-black?style=flat&logo=github)](https://github.com/KiranNaseer-AI)
[![ORCID](https://img.shields.io/badge/ORCID-A6CE39?style=flat&logo=orcid&logoColor=white)](https://orcid.org/0009-0005-5129-8155)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
