##############################################################################
# PAISS 2018: Understanding image retrieval representations                  #
# NLE practical session 02/06/2018                                           #
# WARNING: Do not run "python script.py". Please open a ipython terminal and #
# run blocks of commands one by one.                                         #
##############################################################################

# Run preliminary imports
import numpy as np
from numpy.linalg import norm
import torch
from torch import nn
import json
import pdb
import sys

from datasets import create
from archs import *
from utils.test import q_eval
from utils.tsne import do_tsne

import argparse

parser = argparse.ArgumentParser(description='Demo')

parser.add_argument('--sect', type=str, default='init', required=False, help='Selects the section to run')
parser.add_argument('--qidx', type=int, required=False, help='Query index')
parser.add_argument('--hide_tsne', dest='show_tsne', action='store_false', default=True, help='Skips the TSNE computation')
args = parser.parse_args()

np.random.seed(0)

#############################
#### Section 1: Training ####
#############################

# 1a: AlexNet architecture
# 1b: Finetuning on Landmarks
# 1c: Generalized Mean Poolimg (GeM)
# 1d: ResNet18 architecture
# 1e: PCA
# 1f: Triplet loss and training for retrieval
# 1g: Data augmentation
# 1h: Multi-resolution
# 1i: Improved architectures

# create Oxford 5k database
dataset = create('Oxford')

# get the label vector
labels = dataset.get_label_vector()
classes = dataset.get_label_names()

# load the dictionary of the available models and features
with open('data/models.json', 'r') as fp:
    models_dict = json.load(fp)

# load the necessary models and data
if args.sect in ['2a', '2b', '2c']:
    # load models:
    model1 = resnet50_rank()
    model1.eval()
    model2 = resnet50_rank_DA()
    model2.eval()

if args.sect.startswith('2'):
    dfeats1 = np.load(models_dict['resnet50-rnk-lm-gem']['dataset'])
    qfeats1 = np.load(models_dict['resnet50-rnk-lm-gem']['queries'])
    dfeats2 = np.load(models_dict['resnet50-rnk-lm-gem-da']['dataset'])
    qfeats2 = np.load(models_dict['resnet50-rnk-lm-gem-da']['queries'])


if args.qidx is not None and args.qidx > 54:
    print ('Incorrect query index. Please choose an id from 0 to 54. Exiting...')
    sys.exit()

print('Running section {}\n'.format(args.sect))

if args.sect == 'dataset':
    print(classes.tolist())

    # visualize Oxford queries
    dataset.vis_queries()


elif args.sect == '1a':
    #### 1a: AlexNet architecture ####
    model_1a = alexnet_imagenet()
    print(model_1a)
    input("Check session.py. Press Enter to continue...")
    # Q: The original AlexNet model is trained for classification on ImageNet dataset, that contains 1000 image classes. What is the number of dimensions of the output of the model? What does each dimension represents?

    dfeats = np.load(models_dict['alexnet-cls-imagenet-fc7']['dataset'])
    print(norm(dfeats[:10], axis=1))
    print(dfeats.shape)
    input("Check session.py. Press Enter to continue...")
    # Q: What does each line of the matrix feats represent? Where does the dimension of these lines comes from and how do we extract these features?
    # Hint: uncomment and run the following command
    # model_1a_test = alexnet_imagenet_fc7(); print(model_1a_test)

    # visualize top results for a given query
    q_idx = args.qidx if args.qidx is not None else 0

    dataset.vis_top(dfeats, q_idx, ap_flag=True)

    if args.show_tsne:
        # run t-SNE
        do_tsne(dfeats, labels, classes, sec='1a')
        # Q: What can be observe from the t-SNE visualization? Which classes 'cluster' well? Which do not?



elif args.sect == '1b':

    #### 1b: Finetuning on Landmarks ####
    model_1b = alexnet_lm()
    print(model_1b)
    input("Check session.py. Press Enter to continue...")
    # Q: Why do we change the last layer of the AlexNet architecture? How do we initialize the layers of model_1b for finetuning?

    dfeats = np.load(models_dict['alexnet-cls-lm-fc7']['dataset'])
    qfeats = np.load(models_dict['alexnet-cls-lm-fc7']['queries'])
    q_idx = args.qidx if args.qidx is not None else 0
    dataset.vis_top(dfeats, q_idx, q_feat=qfeats[q_idx], ap_flag=True)

    if args.show_tsne:
        # run t-SNE
        do_tsne(dfeats, labels, classes, sec='1b')
        # Q: How does the visualization change after finetuning? What about the top results?

    # question on how the architecture demands the resize of the input images (specifically, the fully connected layers) ##########




elif args.sect == '1c':

    #### 1c: Generalized Mean Pooling (GeM) ####
    model_1c = alexnet_GeM()
    print(model_1c)
    input("Check session.py. Press Enter to continue...")
    # Q: For this model, we remove all fully connected layers (classifier layers) and replace the last max pooling layer by an aggregation pooling layer (more details about this layer in the next subsection)

    dfeats = np.load(models_dict['alexnet-cls-lm-gem']['dataset'])
    qfeats = np.load(models_dict['alexnet-cls-lm-gem']['queries'])
    print(dfeats.shape)
    input("Check session.py. Press Enter to continue...")
    # Q: Why does the size of the feature representation changes? Why does the size of the feature representation is important for a image retrieval task?
    q_idx = args.qidx if args.qidx is not None else 0
    dataset.vis_top(dfeats, q_idx, q_feat=qfeats[q_idx], ap_flag=True)

    if args.show_tsne:
        do_tsne(dfeats, labels, classes, sec='1c')
        # Q: How does the aggregation layer changes the t-SNE visualization? Can we see some structure in the clusters of similarly labeled images?



elif args.sect == '1d':

    #### 1d: Resnet18 architecture with GeM pooling ####
    model_0 = resnet18()
    model_1d = resnet18_GeM()
    print(model_0.adpool)
    print(model_1d.adpool)
    input("Check session.py. Press Enter to continue...")
    # Q: Why do we change the average pooling layer of the original Resnet18
    # architecture for a generalized mean pooling? What operation is the layer
    # model_1d.adpool doing?
    # Hint: You can see the code of the generalized mean pooling in file pooling.py

    # load oxford features from ResNet18 model
    dfeats = np.load(models_dict['resnet18-cls-lm-gem']['dataset'])
    qfeats = np.load(models_dict['resnet18-cls-lm-gem']['queries'])
    print(norm(dfeats[:10], axis=1))
    print(dfeats.shape)
    # visualize top results for a given query index
    q_idx = args.qidx if args.qidx is not None else 0
    dataset.vis_top(dfeats, q_idx, q_feat=qfeats[q_idx], ap_flag=True)

    if args.show_tsne:
        do_tsne(dfeats, labels, classes, sec='1d')
        # Q: How does this model compare with model 1c, that was trained in the same dataset for the same task? How does is compare to the finetuned models of 1b?



elif args.sect == '1e':

    #### 1e: PCA Whitening ####

    # We use a PCA learnt on landmarks to whiten the output features of 'resnet18-cls-lm-gem'
    dfeats = np.load(models_dict['resnet18-cls-lm-gem-pcaw']['dataset'])
    qfeats = np.load(models_dict['resnet18-cls-lm-gem-pcaw']['queries'])
    if args.show_tsne:
        do_tsne(dfeats, labels, classes, sec='1e-1')
        # run t-SNE including unlabeled images
        do_tsne(dfeats, labels, classes, sec='1e-2', show_unlabeled=True)
        # Q: What can we say about the separation of data when included unlabeled images? And the distribution of the unlabeled features? How can we train a model to separate labeled from unlabeled data?



elif args.sect == '1f':

    #### 1f: Finetuning on Landmarks for retrieval ####
    # Now we learn the architecture presented in 1e in and end-to-end manner for the retrieval task
    # The architecture includes a FC that replaces the PCA projection
    dataset.vis_triplets()

    dfeats = np.load(models_dict['resnet18-rnk-lm-gem']['dataset'])
    qfeats = np.load(models_dict['resnet18-rnk-lm-gem']['queries'])
    q_idx = args.qidx if args.qidx is not None else 0
    dataset.vis_top(dfeats, q_idx, q_feat=qfeats[q_idx], ap_flag=True)
    if args.show_tsne:
        do_tsne(dfeats, labels, classes, sec='1f-1')
        do_tsne(dfeats, labels, classes, sec='1f-2', show_unlabeled=True)
        # Q: Compare the plots with unlabeled data of the model trained for retrieval (with triplet loss) and the model trained for classification of the previous subsection. How does it change?



elif args.sect == '1g':

    #### 1g: Data augmentation ####
    # This model has been trained the following data augmentation:
    # cropping, pixel jittering, rotation, tilting
    dfeats = np.load(models_dict['resnet18-rnk-lm-gem-da']['dataset'])
    qfeats = np.load(models_dict['resnet18-rnk-lm-gem-da']['queries'])
    q_idx = args.qidx if args.qidx is not None else 0
    dataset.vis_top(dfeats, q_idx, q_feat=qfeats[q_idx], ap_flag=True)

    if args.show_tsne:
        do_tsne(dfeats, labels, classes, sec='1g')



elif args.sect == '1h':
    print('HOLA!')
    #### 1h: Multi-resolution ####
    # Using the same model as the one in sect-1g, we extract features at 4
    # different resolutions and average the outputs
    dfeats = np.load(models_dict['resnet18-rnk-lm-gem-da-mr']['dataset'])
    qfeats = np.load(models_dict['resnet18-rnk-lm-gem-da-mr']['queries'])
    q_idx = args.qidx if args.qidx is not None else 0
    dataset.vis_top(dfeats, q_idx, q_feat=qfeats[q_idx], ap_flag=True)

    if args.show_tsne:
        do_tsne(dfeats, labels, classes, sec='1h')



elif args.sect == '1i':

    # 1i: Improved architecture
    # Finally, we upgrade the backbone architecture to Resnet50
    dfeats = np.load(models_dict['resnet50-rnk-lm-gem-da-mr']['dataset'])
    qfeats = np.load(models_dict['resnet50-rnk-lm-gem-da-mr']['queries'])
    q_idx = args.qidx if args.qidx is not None else 0
    dataset.vis_top(dfeats, q_idx, q_feat=qfeats[q_idx], ap_flag=True)

    if args.show_tsne:
        do_tsne(dfeats, labels, classes, sec='1i')



#######################################
####      Section 2: Testing       ####
#######################################
elif args.sect == '2a':
    #### Section 2a: Robustness to input transformations
    q_idx = args.qidx if args.qidx is not None else 0

    q_feat1 = q_eval(model1, dataset, q_idx)
    dataset.vis_top(dfeats1, q_idx, q_feat1, ap_flag=True)

    # Flipping the query image
    q_feat1_flip = q_eval(model1, dataset, q_idx, flip=True)
    dataset.vis_top(dfeats1, q_idx, q_feat1_flip, ap_flag=True)
    # Q1: What is the impact of flipping the query image?

    # Standard trick: aggregate both no-flipped and flipped representations
    q_feat1_new = (q_feat1 + q_feat1_flip)
    q_feat1_new = q_feat1_new / norm(q_feat1_new) # Don't forget to l2-normalize again :)
    dataset.vis_top(dfeats1, q_idx, q_feat1_new, ap_flag=True)

    # Rotating the query image
    q_feat1_rot = q_eval(model1, dataset, q_idx, rotate=5.)
    dataset.vis_top(dfeats1, q_idx, q_feat1_rot, ap_flag=True)

    q_feat2_rot = q_eval(model2, dataset, q_idx, rotate=5.)
    dataset.vis_top(dfeats2, q_idx, q_feat2_rot, ap_flag=True)
    # Q2: Change the rotation value (in +/- degrees). What is the impact of rotating it? Up to which degree of rotation is the result stable? How does the models (model1 trained without image rotation, model2 trained with) compare?


elif args.sect == '2b':
    #### Section 2b: Queries with multi-scale features
    q_idx = args.qidx if args.qidx is not None else 0

    # Extract features using a single input scale: 800px
    q_feat = q_eval(model1, dataset, q_idx)
    dataset.vis_top(dfeats1, q_idx, q_feat, ap_flag=True)

    # Aggregate features extracted at several input sizes: [600, 800, 1000, 1200]
    q_feat_mr = q_eval(model1, dataset, q_idx, scale=2)
    dataset.vis_top(dfeats1, q_idx, q_feat_mr, ap_flag=True)
    # Q: What is the impact of using more scales?


elif args.sect == '2c':
    #### Section 2c: Robustness to resolution changes
    q_idx = args.qidx if args.qidx is not None else 0

    # Extract features using a larger input scale: 1200px
    q_feat = q_eval(model1, dataset, q_idx, scale=1.5)
    dataset.vis_top(dfeats1, q_idx, q_feat, ap_flag=True)
    # Q: Resize the image by a factor. What is the impact of resizing it, especially to very low resolution?


elif args.sect == '2d':
    ## Subsection 2d: Robustness to compression (using PQ)
    q_idx = args.qidx if args.qidx is not None else 0
    dataset.vis_top(dfeats2, q_idx, ap_flag=True)

    m = 256      # number of subquantizers
    n_bits = 8   # bits allocated per subquantizer

    feats_train = np.load(models_dict['resnet50-rnk-lm-gem-da']['training'])
    dataset.pq_train(feats_train, m, n_bits)

    # dataset to encode
    dataset.pq_add(feats)

    # search:
    dataset.vis_top(dfeats2, q_idx, pq_flag=True, ap_flag=True)
    # Q1: How much memory (in bytes) is needed to store the compressed representation?
    # Q2: What is the compression ratio?
    # Q3: How did the compression affect the retrieval results?
    # Q4: Change the values and m & n_bit and observe the change in retrieval performance.

elif args.sect == '2e':
    ## Subsection 2e: Average query expansion
    q_idx = args.qidx if args.qidx is not None else 0

    dataset.vis_top(dfeats2, q_idx, q_feat=qfeats2[q_idx], nqe=3, ap_flag=True)
    # nqe is the number of database items with which to expand the query.
    # Q1: What is the impact of using different values of nqe?

elif args.sect == '2f':
    ## Subsection 2f: alpha query expansion
    q_idx = args.qidx if args.qidx is not None else 0

    dataset.vis_top(dfeats2, q_idx, q_feat=qfeats2[q_idx], nqe=5, aqe=3.0, ap_flag=True)
    # aqe is the value of alpha applied for alpha query expansion.
    # Q1: How should nqe be chosen? Hint: What is the impact of low prec@K (where K is equivalent to nqe) on aqe?
    # Q2: What is the impact of using different values of nqe, aqe?

elif args.sect == '2g':
    ## Subsection 2g: Diffusion
    q_idx = args.qidx if args.qidx is not None else 0

    dataset.vis_top(dfeats2, q_idx, q_feat=qfeats2[q_idx], dfs='it:int20', ap_flag=True)
    # Parameters for dfs are passed as strings with datatypes indicated. The default parameter string is:
    #    'alpha:float0.99_it:int20_tol:float1e-6_gamma:float3_ks:100-30_trunc:bool_bsize:int100000_fsr:bool_IS:bool_wgt:bool_bs:bool_reg:bool_split:int0_gmp:bool'
    #    strings passed to the dfs parameter overwrite the default parameters

    # Q1: The affinity matrix is computed using the similarity measure s = <f_i, f_j>^alpha, where 0 < alpha <= 1.0. Use dfs='alpha:float<alpha>' for different values of alpha. What is the impact of changing it? E.g:
    dataset.vis_top(dfeats2, q_idx, q_feat=qfeats2[q_idx], dfs='alpha:float0.8', ap_flag=True)

    # Q2: k_q is the number of database items to use for diffusion. Use dfs='ks:100-<k_q>' for different values of k_q. What is the impact of changing it? E.g:
    dataset.vis_top(dfeats2, q_idx, q_feat=qfeats2[q_idx], dfs='ks:100-5', ap_flag=True)

    # Q3: trunc is the number of sub-rows and columns to use for diffusion. Use dfs='trunc:int<trunc>' for different values of trunc. What is the impact of changing it? E.g:
    dataset.vis_top(dfeats2, q_idx, q_feat=qfeats2[q_idx], dfs='trunc:int2000', ap_flag=True)
    # Q4: What is the maximum value of trunc and what case does it generalize to?

else:
    print ('Incorrect section name. Please choose a section between 1[a-i] or 2[a-g]. Example: python session.py --sect 1c')
