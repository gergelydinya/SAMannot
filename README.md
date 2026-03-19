<a href="https://arxiv.org/abs/2601.11301"><img src="https://img.shields.io/badge/arXiv-2503.11651-b31b1b" alt="arXiv"></a>
<a href="https://samannot.github.io/"> <img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href="https://www.youtube.com/watch?v=j5TVcVeuVZM"> <img src="https://img.shields.io/badge/demo_video-youtube-red" alt="Demo video"></a>
<a href="https://www.youtube.com/watch?v=X8AkZTCYWQo"> <img src="https://img.shields.io/badge/demo_video-youtube-red" alt="Install video"></a>

# SAMannot

SAMannot is a versatile video annotation tool built on top of Meta's Segment Anything Model (SAM2). It helps you create high-quality segmentation masks across video frames with minimal user interaction.


## Installation Guide

This guide explains how to set up the environment, download required checkpoints, and launch the app

### Prerequisites

- **Conda** (Anaconda or Miniconda)
- **Git** (optional, if you clone this repo)
- **Python 3.10** (the Conda env will install this)
- **CUDA-enabled GPU** and the correct drivers for faster inference
- **nvidia-cuda-toolkit** installed (optional, recommended)

### 1) Clone repository
```bash
git clone https://github.com/gergelydinya/SAMannot.git
cd SAMannot
```
### 2) Create and activate the Conda environment

```bash
# From the project root
conda create -n samannot python=3.10 -y
conda activate samannot
pip install -r requirements.txt
cd sam2
pip install -e .
cd ..
# Match to your CUDA version 12.8 ~ cu128
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
```

### 3) Download checkpoints
```bash
cd checkpoints
bash download_chckpts.sh
```

### 4) Run the software
```bash
conda activate samannot
python main.py
```

## Evaluation
```bash
cd eval
python eval_fast.py --gt <ground truth folder> --pred <prediction folder>
```

## Best practices and practical tips
The typical workflow the software is designed for is that the user proceeds block by block and moves forward without going back.
It is also possible to load an earlier block, but in that case you should expect some time overhead, as the software needs to load the frames starting from the beginning of the video.

Choose your block size based on your computer’s resources and the complexity of the data.
As a starting point, for an average video, we recommend a block size of 100--150 frames.

## Citation

If you use **SAMannot**, please cite our paper:

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
