# Sub-Millisecond Solutions to Category-Level Shape and Pose Estimation
[[Paper](TODO)] [[Video](TODO)]

Official Julia implementation of "A Sub-Millisecond Solver for Category-Level Object Shape and Pose Estimation"
by Lorenzo Shaikewitz, Tim Nguyen, and Luca Carlone

**TODO: nice figure / animation here (see my website)**

## Quick Start
TODO:
- Data folder
- Julia environment

## Reproducing Results
All methods: recommending one method at a time to prevent from going crazy.

- Synthetic:
- CAST: give link to keypoints (detector not provided)
- NOCS: give link to keypoints, YOLO detector pt files (yolov8 github)
- ApolloScape: all from GSNet

ApolloScape / GSNet Install:
1. Install detectron2
```bash
# create virtual environment
python3 -m venv .env
# install pytorch: https://pytorch.org/get-started/locally/
# build detectron2 from source (https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md)
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
If you run into trouble with the last step, try updating setuptools first:
```
# update wheel (optional, may not be necessary)
pip install -U pip setuptools wheel
```
2. Install GSNet
```bash
# clone the repo
git clone https://github.com/lkeab/gsnet.git
# go to source code
cd gsnet/reference_code/GSNet-release
```
LINK TO PRETRAINED WEIGHTS IS BROKEN

## References


## BibTeX
