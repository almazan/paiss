# test.py: eval query image
import torch
from math import floor
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Normalize, Resize, RandomHorizontalFlip, RandomRotation, Compose

def q_eval(net, dataset, q_idx, flip=False, rotate=False, scale=1):
    # load query image
    q_im = dataset.get_image(q_idx)
    q_size = q_im.size

    # list of transformation lists
    trfs_chains = [[]]
    if rotate:
        eps = 1e-6
        trfs_chains[0] += [RandomRotation((rotate-eps,rotate+eps))]
    if flip:
        trfs_chains[0] += [RandomHorizontalFlip(1)]
    if scale == 0: # AlexNet asks for resized images of 224x224
        edge_list = [224]
        resize_list = [Resize((edge,edge)) for edge in edge_list]
    elif scale == 1:
        edge_list = [800]
        resize_list = [lambda im: imresize(im, edge) for edge in edge_list]
    elif scale == 1.5:
        edge_list = [1200]
        resize_list = [lambda im: imresize(im, edge) for edge in edge_list]
    elif scale == 2: # multiscale
        edge_list = [600,800,1000,1200]
        resize_list = [lambda im: imresize(im, edge) for edge in edge_list]
    else:
        raise ValueError()

    if len(resize_list) == 1:
        trfs_chains[0] += resize_list
    else:
        add_trf(trfs_chains, resize_list )

    # default transformations
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for chain in trfs_chains:
        chain += [ToTensor(), Normalize(mean, std)]

    net = net.eval()
    q_feat = torch.zeros( (len(trfs_chains), net.out_features) )
    print ('Computing the forward pass and extracting the image representation...')
    for i in range(len(trfs_chains)):
        q_tensor = Compose(trfs_chains[i])(q_im)
        q_feat[i] = net.forward(q_tensor.view(1,q_tensor.shape[0],q_tensor.shape[1],q_tensor.shape[2]))
    return F.normalize(q_feat.mean(dim=0), dim=0).detach().numpy()


def add_trf(trfs_chains, new_trfs):
    n_chains = len(trfs_chains)
    for trf in new_trfs:
        for i in range(n_chains):
            trfs_chains.append(trfs_chains[i] + [trf])
    return

def imresize(im, maxedge):
    ''' creates image in a different size, where the aspect ratio as maintainted max max(height,width)=maxedge
    '''
    h,w = im.size
    if h<w:
        return im.resize((int(h/w*maxedge), maxedge))
    else:
        return im.resize((maxedge, int(h/w*maxedge)))
