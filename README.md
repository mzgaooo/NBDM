# Beyond Linear Dynamics: Neural Bilinear Dynamical Models for Time Series Forecasting

This repository provides the PyTorch implementation of:

> **Beyond Linear Dynamics: Neural Bilinear Dynamical Models for Time Series Forecasting**  
> Mengzhou Gao, Huangqian Yu, Pengfei Jiao  
> Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), 2026  
> [Paper](https://doi.org/10.1145/3770855.3818066)

---

## Introduction

Time series forecasting is a fundamental yet challenging problem due to the complex nonlinear dynamics of real-world systems, which are difficult to capture using linear or Koopman-based approximations.

In this work, we propose the **Neural Bilinear Dynamical Model (NBDM)**, a novel framework that explicitly models nonlinear system dynamics through bilinear evolution in the Koopman latent space.

Our method:

- a bilinear dynamic propagation module to capture multiplicative interactions between latent system states and control inputs
- a learnable error compensation term to correct residual modeling errors arising from latent representations and bilinear approximations
- a memory-enhanced feedback controller to infer latent control signals from historical state-control interactions when external control inputs are unavailable.

---

## ⚙️ Dependencies

```text
numpy<2.0
torch
pandas
scipy
scikit-learn
matplotlib
```

---

## 🚀 Quick Start

### 1. Data Preparation

**Given Control Inputs**

- `PeMS04`
- `PeMS08`

The PeMS04 and PeMS08 datasets are available at: https://github.com/guoshnBJTU/ASTGNN

**Missing Control Inputs**

- `Temperature`
- `Seoul PM2.5`
- `PeMS-Bay`

The Temperature and Seoul PM2.5 datasets are available at: https://github.com/QipengW/SF-GDE/

The PeMS-Bay dataset is available at: https://github.com/george-j-ste/Augmented-PEMS-BAY/tree/main

### 2. Run NBDM

To run NBDM on the PeMS04 dataset:

```bash
cd "Given control input"
python run.py --config ./config_file/PEMS04.conf
```

---

## 📖 Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{gao2026beyond,
  title     = {Beyond Linear Dynamics: Neural Bilinear Dynamical Models for Time Series Forecasting},
  author    = {Gao, Mengzhou and Yu, Huangqian and Jiao, Pengfei},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
  year      = {2026},
  doi       = {10.1145/3770855.3818066},
  url       = {https://doi.org/10.1145/3770855.3818066}
}
```
