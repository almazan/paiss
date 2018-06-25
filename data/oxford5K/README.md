Download the Oxford 5k images in this folder by running the cmd:
wget www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz -O images.tgz

Extract the images in a folder /jpg by running the cmd:
mkdir jpg && tar -xzf images.tgz -C jpg

Download the Oxford 5k groundtruth files in this folder by running the cmd:
wget www.robots.ox.ac.uk/~vgg/data/oxbuildings/gt_files_170407.tgz -O gt_files.tgz

Extract the files in a folder /lab by running the cmd:
mkdir lab && tar -xzf gt_files.tgz -C lab
