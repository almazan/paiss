# PAISS18
NLE practical session for PAISS 2018

## Installation

### Linux / MacOS

First, download and install appropriate version of miniconda using instructions the instructions for [MacOS](https://conda.io/docs/user-guide/install/macos.html) or [Linux](https://conda.io/docs/user-guide/install/linux.html).

Then run the following commands:

```
conda install numpy matplotlib ipython
conda install pytorch torchvision faiss-cpu -c pytorch
```

On MacOS there’s a bug for faiss related to libomp (https://github.com/facebookresearch/faiss/issues/485): run “brew install libomp”  (see https://brew.sh/ to install brew) to resolve this bug.

### Windows

Install anaconda on windows (launch .exe file downloaded from the [conda website](https://conda.io/docs/user-guide/install/windows.html)). Has to be python 3 (pytorch doesn’t support 2.7 on windows)

In the anaconda prompt, run:

```
conda create -n pytorch
activate pytorch
conda install pytorch-cpu -c pytorch
pip install torchvision --no-deps
conda install pillow
```

### Donwload the necessary files

Oxford dataset:
```
wget www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz -O images.tgz
mkdir data/oxford5k/jpg && tar -xzf images.tgz -C data/oxford5k/jpg
wget www.robots.ox.ac.uk/~vgg/data/oxbuildings/gt_files_170407.tgz -O gt_files.tgz
mkdir data/oxford5k/lab && tar -xzf gt_files.tgz -C data/oxford5k/lab
```


Features and models:
```
wget ...
tar -xzf features.tgz -C data
wget ...
tar -xzf models.tgz -C data
```


## Demo

Run `python demo.py` and you should see the following ouput:




