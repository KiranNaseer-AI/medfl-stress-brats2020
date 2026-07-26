# MedFL-Stress: A Systematic Robustness Evaluation of Federated Brain Tumor Segmentation under Cross-Hospital MRI Appearance Shift

> **FORGE-BENCH** | Part of the [FORGE Framework](https://github.com/KiranNaseer-AI/KiranNaseer-AI) for Out-of-distribution Robustness, Generalisation and Evaluation

[![Paper](https://img.shields.io/badge/DeCaF_Workshop-MICCAI_2026-blue?style=flat)](https://workshop-decaf.github.io/decaf-2026/)
[![Status](https://img.shields.io/badge/Status-Under_Review-yellow?style=flat)]()
[![Dataset](https://img.shields.io/badge/Dataset-BraTS_2020-green?style=flat)](https://www.med.upenn.edu/cbica/brats2020/data.html)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python&logoColor=white)](https://python.org)

---

## What This Paper Is About

Federated learning (FL) lets hospitals train segmentation models together without sharing patient data. Standard FL evaluation reports one number — **mean Dice across all sites** — and that number can look competitive while one hospital runs far behind the rest.

**MedFL-Stress** is a controlled stress-testing protocol built to expose that failure. We simulate four hospital clients from BraTS 2020, each with a fixed, graded MRI appearance shift (gamma contrast, intensity scale-shift, or noise-plus-blur — see heterogeneity levels H0–H3), and treat **worst-client Dice** and **best–worst gap** as primary metrics rather than supplementary ones.

### Key Finding

The worst-client gap is not primarily a federation problem. Even a **centralised** model trained on all pooled data shows a 0.055 best–worst Dice gap under appearance shift alone. Federation (FedAvg) adds a further 0.030 on top of that. **Appearance shift is the root cause; federation amplifies it.**

We test five methods — Centralised, FedAvg, FedProx, FedBN, and q-FedAvg — across two evaluation regimes (single-seed full-data, and five-seed subsampled) to check whether conclusions hold up under repetition.

---

## Results

### Full-Data Results (single seed, H3, 10 rounds)

| Method | WT | TC | ET | Mean | Worst | Best | Gap |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Centralised | 0.845 | 0.598 | 0.534 | 0.659 | 0.622 | 0.678 | 0.055 |
| FedAvg | 0.827 | 0.587 | 0.506 | 0.640 | 0.592 | 0.676 | 0.085 |
| FedProx (μ=0.01) | 0.785 | 0.553 | 0.478 | 0.605 | 0.560 | 0.654 | 0.094 |
| FedBN | 0.831 | 0.572 | 0.498 | 0.634 | 0.584 | 0.665 | 0.081 |
| **q-FedAvg (q=1)** | 0.830 | 0.589 | 0.510 | **0.643** | **0.596** | 0.677 | 0.081 |

### Multi-Seed Reliability (5 seeds, subsampled, H3)

| Method | Mean Dice | Worst-Client Dice | Gap | p vs. FedAvg |
|---|:---:|:---:|:---:|:---:|
| FedAvg | 0.548 ± 0.015 | 0.502 ± 0.018 | 0.087 ± 0.010 | — |
| FedProx (μ=0.01) | 0.541 ± 0.015 | **0.487 ± 0.018** | 0.103 ± 0.012 | **0.019\*** |
| FedBN | 0.542 ± 0.017 | 0.497 ± 0.033 | 0.086 ± 0.026 | 0.655 |
| **q-FedAvg (q=1)** | **0.555 ± 0.016** | **0.508 ± 0.019** | 0.094 ± 0.010 | 0.436 |

**Three findings hold across all five seeds:**
- The best–worst gap never closes for any method, in any run.
- **q-FedAvg** achieves the best worst-client Dice in both regimes while also holding the highest mean Dice — no accuracy–fairness tradeoff in this setting.
- **FedProx significantly hurts the worst client** (p=0.019, paired t-test vs. FedAvg) — proximal regularisation keeps local updates anchored to a global model already biased toward the clean-domain client, which prevents the hardest client from adapting.

**A result single-seed evaluation would have missed:** FedBN's worst-client Dice looks reasonable on average, but its variance across seeds (CI₉₅ = 0.026) is roughly 2.6× FedAvg's — a reliability gap invisible in any single run. Worst-client *rankings* replicate identically across both evaluation regimes (q-FedAvg best, FedProx worst), but reliability does not, which is why we report worst-client Dice as the primary metric rather than a single-run gap estimate.

---

## Method

- **Model:** 2D U-Net (T1, T1ce, T2, FLAIR input channels), predicting WT / TC / ET per the BraTS protocol
- **Dataset:** BraTS 2020, 369 cases; 55 held out (15%) before partitioning; remaining 314 split across 4 clients (79/79/78/78)
- **Heterogeneity protocol:** each client receives a fixed appearance transform — gamma contrast (Client 2), affine scale-shift (Client 3), noise + blur (Client 4) — graded across four severities H0 (none) to H3 (strong)
- **Federated methods:** FedAvg, FedProx (μ=0.01), FedBN, q-FedAvg (q=1, selected via pilot sweep over q ∈ {0.5, 1, 2, 5})
- **Evaluation regimes:** full-data (single seed, 10 rounds) and subsampled (200 slices/client/round, 20 rounds, 5 seeds: 42, 123, 456, 789, 1001)

**Limitations (from the paper):** four clients, one dataset; 2D slices, not directly comparable to volumetric BraTS benchmarks; the multi-seed reliability check runs on a subsampled regime for tractability; transforms cover appearance variability only, not annotation disagreement, demographic shift, or workflow differences.

---

## Code Availability

Per the paper: **code will be released upon acceptance.** This repository currently hosts the paper record and results; the training/evaluation pipeline will be added once review completes.

---

## Citation

```bibtex
@inproceedings{naseer2026medflstress,
  title     = {MedFL-Stress: A Systematic Robustness Evaluation of Federated
               Brain Tumor Segmentation under Cross-Hospital MRI Appearance Shift},
  author    = {Naseer, Kiran and Butt, Naveed Anwar},
  booktitle = {DeCaF Workshop, MICCAI 2026},
  year      = {2026},
  note      = {Under review}
}
```

*(arXiv preprint 2605.09025 currently reflects an earlier version of this work under a different title; an updated version will be posted separately.)*

---

## FORGE Framework

This repository is the **FORGE-BENCH** component of the FORGE research programme — a unified framework for out-of-distribution robustness, generalisation, and evaluation in federated and multimodal AI systems.

| Component | Role | Status |
|---|---|---|
| **FORGE-BENCH** | Stress-test benchmarking — federated medical imaging | This repo · DeCaF Workshop, MICCAI 2026 · Under review |
| **FORGE-DIAG** | Diagnosing VLM instability under physical domain shift | [multimodal-domain-shift-vlm](https://github.com/KiranNaseer-AI/multimodal-domain-shift-vlm) · IEEE TMM · In preparation |
| **FORGE-EVAL** | Worst-client robustness in federated systems | [FM-Fairness-Paradox](https://github.com/KiranNaseer-AI/FM-Fairness-Paradox) · FL@FM–IJCAI 2026, Accepted |
| **FORGE-ADAPT** | Adaptive gradient-signal stage switching | Thesis Ch. 4 · In progress |

---

## Author

**Kiran Naseer** — PhD Researcher, University of Gujrat, Pakistan
Co-supervised by Dr. Naveed Anwar Butt

[![Google Scholar](https://img.shields.io/badge/Google_Scholar-4285F4?style=flat&logo=google-scholar&logoColor=white)](https://scholar.google.com/citations?user=Ek9e3qwAAAAJ&hl=en)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kiran-naseer)
[![GitHub](https://img.shields.io/badge/GitHub-KiranNaseer--AI-black?style=flat&logo=github)](https://github.com/KiranNaseer-AI)
[![ORCID](https://img.shields.io/badge/ORCID-A6CE39?style=flat&logo=orcid&logoColor=white)](https://orcid.org/0009-0005-5129-8155)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
