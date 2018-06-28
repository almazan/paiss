# PAISS18
NLE practical session for [PAISS 2018](https://project.inria.fr/paiss/)

## Installation

### Linux / MacOS

First, download and install the appropriate version of miniconda following the instructions for [MacOS](https://conda.io/docs/user-guide/install/macos.html) or [Linux](https://conda.io/docs/user-guide/install/linux.html).

Then run the following commands:

```
conda install numpy matplotlib ipython scikit-learn
conda install pytorch torchvision faiss-cpu -c pytorch
```

On MacOS there’s a bug for faiss related to libomp (https://github.com/facebookresearch/faiss/issues/485): run “brew install libomp”  (see https://brew.sh/ to install brew) to resolve this bug.

### Windows

Install anaconda on windows (launch .exe file downloaded from the [conda website](https://conda.io/docs/user-guide/install/windows.html)). It has to be python 3 (pytorch doesn’t support 2.7 on windows)

In the anaconda prompt, run:

```
conda create -n pytorch
activate pytorch
conda install pytorch-cpu -c pytorch
pip install torchvision --no-deps
conda install pillow
```

**NOTE: _The FAISS package is not supported on Windows._ Participants with Windows machines must follow the product quantization exercise with their neighbours.**

### Download the dataset and models

You will need to download 4 files:
- oxbuild_images.tgz (1.8GB)
- gt_files_170407.tgz (280KB)
- features.tgz (579MB)
- models.tgz (328MB)

and store them in the appropriate paths.

_Note:_ All paths in this section are relative to the root directory of this repository.

**Oxford dataset**

On Linux/MacOS, execute the following:
```
wget www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz -O images.tgz
mkdir -p data/oxford5k/jpg && tar -xzf images.tgz -C data/oxford5k/jpg
wget www.robots.ox.ac.uk/~vgg/data/oxbuildings/gt_files_170407.tgz -O gt_files.tgz
mkdir -p data/oxford5k/lab && tar -xzf gt_files.tgz -C data/oxford5k/lab
```

On Windows, perform the following:
- Download www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz
- create directory data/oxford5k/jpg/
- uncompress oxbuild_images.tgz and store in data/oxford5k/jpg/
- Download www.robots.ox.ac.uk/~vgg/data/oxbuildings/gt_files_170407.tgz
- create directory data/oxford5k/lab/
- uncompress gt_files_170407.tgz and store in data/oxford5k/lab/

**Features and models**

On Linux/MacOS, execute the following:
```
wget https://www.dropbox.com/s/gr404xlfr4021pw/features.tgz?dl=1 -O features.tgz
tar -xzf features.tgz -C data
wget https://www.dropbox.com/s/mr4risqu7t9neel/models.tgz?dl=1 -O models.tgz
tar -xzf models.tgz -C data
```
On Windows, perform the following:
- Download https://www.dropbox.com/s/gr404xlfr4021pw/features.tgz?dl=1
- Uncompress features.tgz and store in data/
- Download https://www.dropbox.com/s/mr4risqu7t9neel/models.tgz?dl=1
- Uncompress models.tgz and store in data/


## Demo

Run `python demo.py --qidx 42 --topk 5` and you should see the following ouput:

![Output](https://www.dropbox.com/s/pgboc4yrehvdsh7/out.png?raw=1)
