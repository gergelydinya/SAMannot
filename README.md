# SAMannot — Installation Guide

This guide explains how to set up the environment, download required checkpoints, and launch the app

## Prerequisites

- **Conda** (Anaconda or Miniconda)
- **Git** (optional, if you clone this repo)
- **Python 3.10** (the Conda env will install this)
- **CUDA-enabled GPU** and the correct drivers for faster inference

---
## 1) Clone repository
```bash
git clone https://github.com/gergelydinya/SAMAnnot.git
cd SAMAnnot
```
## 2) Create and activate the Conda environment

```bash
# From the project root
conda create -n samannot python=3.10 -y
conda activate samannot
pip install -r requirements.txt
cd sam2
pip install -e .
cd..
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

## 3) Download checkpoints
```bash
cd checkpoints
./download_chckpts.sh
```

## 4) Run the software
```bash
conda activate 
python main.py
```