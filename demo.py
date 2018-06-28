##############################################################################
# PAISS 2018: Understanding image retrieval representations                  #
# NLE practical session 02/07/2018                                           #
# DEMO                                                                       #
##############################################################################

# Run preliminary imports
import numpy as np
from datasets import create
from archs import *
from utils.tsne import do_tsne
from PIL import Image
import utils.transforms as trf
import sys
import os.path as osp

import argparse
parser = argparse.ArgumentParser(description='Demo')

parser.add_argument('--qidx', type=int, default=0, required=False, help='Query index')
parser.add_argument('--topk', type=int, default=5, required=False, help='Showing top-k results')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load the Oxford5k database
dataset = create('Oxford')
print (dataset)

# initialize architecture and load weights
print ('Loading the model...')
model = resnet50_rank_DA().to(device)
model.eval()
print ('Done\n')

# load the precomputed dataset features
d_feats_file = 'data/features/resnet50-rnk-lm-da_ox.npy'
try:
    d_feats = np.load(d_feats_file)
except OSError as e:
    print ('ERROR: File {} not found. Please follow the instructions to download the pre-computed features.'.format(d_feats_file))
    sys.exit()

# Load the query image
img = Image.open(dataset.get_query_filename(args.qidx))
# Crop the query ROI
img = img.crop(tuple(dataset.get_query_roi(args.qidx)))
# Apply transformations
img = trf.resize_image(img, 800)
I = trf.to_tensor(img)
I = trf.normalize(I, dict(rgb_means=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
I = I.unsqueeze(0).to(device)
# Forward pass to extract the features
with torch.no_grad():
    print ('Extracting the representation of the query...')
    q_feat = model(I).numpy()
print ('Done\n')

# Rank the database and visualize the top-k most similar images in the database
dataset.vis_top(d_feats, args.qidx, q_feat=q_feat, topk=args.topk, out_image_file='out.png')
