# Category-Level Shape and Pose Estimation in Less Than a Millisecond
[[Paper](TODO)] [[Video](TODO)]

Official Julia implementation of "Category-Level Object Shape and Pose Estimation in Less Than a Millisecond"
by Lorenzo Shaikewitz, Tim Nguyen, and Luca Carlone

  One Run                  |  Multiple Minima
:-------------------------:|:-------------------------:
![](assets/scf_oneiter.gif)|![](assets/scf_twomins.gif)


This repository contains the *solver*. We open-source keypoint detection and training [here]().

## Quick Start
First, make sure you have [Julia installed](https://julialang.org/install/). This repository was tested with v1.11.6. Then, clone the repository and follow the directions below. We assume you are in the folder repo.
1. Clone this repository
```shell
git clone https://github.com/lopenguin/nepv.git
cd nepv
```
2. Manually install custom dependencies:
```shell
julia --project
using Pkg
Pkg.add("https://github.com/lopenguin/SimpleRotations.jl")
Pkg.add("https://github.com/lopenguin/TSSOS")
Pkg.instantiate()
```
TODO: TEST THIS

At this point, you can run the solver by calling:
```shell
julia --project scripts/demo_pace.jl
```
Or in the Julia REPL
```julia-repl
julia> include("scripts/demo_pace.jl")
```


TODO:
- Data folder (for reproducing results!)
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
```
@misc{Coming soon!}
```