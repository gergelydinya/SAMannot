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

## Citation

If you use **SAMannot** in your research, please cite our paper:

```bibtex
@misc{samannot,
  title={SAMannot: A Memory-Efficient, Local, Open-source Framework for Interactive Video Instance Segmentation based on SAM2},
  author={Gergely Dinya and Andr{\'a}s Gelencs{\'e}r and Krisztina Kup{\'a}n and Clemens K{\"u}pper and Krist{\'o}f Karacs and Anna Gelencs{\'e}r-Horv{\'a}th},
  year={2026},
  eprint={2601.11301},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2601.11301},
}
