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

from datasets import create
from archs import *
#from utils.test import q_eval TODO: uncomment this when test.py is commited
from utils.tsne import do_tsne

np.random.seed(0)

#### Section 1: Training ####
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
print(dataset)

# get the label vector
labels = dataset.get_label_vector()
classes = dataset.get_label_names()
print(classes.tolist())

# visualize Oxford queries
dataset.vis_queries()


# load the dictionary of the available models and features
with open('data/models.json', 'r') as fp:
    models = json.load(fp)


#### 1a: AlexNet architecture ####
model_1a = alexnet_imagenet()
print(model_1a)
# Q: The original AlexNet model is trained for classification on ImageNet dataset, that contains 1000 image classes. What is the number of dimensions of the output of the model? What does each dimension represents?

dfeats = np.load(models['alexnet-cls-imagenet-fc7']['dataset'])
qfeats = np.load(models['alexnet-cls-imagenet-fc7']['queries'])
print(norm(dfeats[:10], axis=1))
print(dfeats.shape)
# Q: What does each line of the matrix feats represent? Where does the dimension of these lines comes from and how do we extract these features?
# Hint: uncomment and run the following command
# model_1a_test = alexnet_imagenet_fc7(); print(model_1a_test)

# visualize top results for a given query
q_idx = 0 # TODO: find good query candidates

dataset.vis_top(dfeats, q_idx, q_feat=qfeats[qidx])

# run t-SNE
do_tsne(feats, labels, classes, sec='1a')
# Q: What can be observe from the t-SNE visualization? Which classes 'cluster' well? Which do not?




#### 1b: Finetuning on Landmarks ####
model_1b = alexnet_lm()
print(model_1b)
# Q: Why do we change the last layer of the AlexNet architecture? How do we initialize the layers of model_1b for finetuning?

dfeats = np.load(models['alexnet-cls-lm-fc7']['dataset'])
qfeats = np.load(models['alexnet-cls-lm-fc7']['queries'])
dataset.vis_top(dfeats, q_idx, q_feat=qfeats[qidx])
# run t-SNE
do_tsne(dfeats, labels, classes, sec='1b')
# Q: How does the visualization change after finetuning? What about the top results?

# question on how the architecture demands the resize of the input images (specifically, the fully connected layers) ##########





#### 1c: Generalized Mean Pooling (GeM) ####
model_1c = alexnet_GeM()
print(model_1c)
# Q: For this model, we remove all fully connected layers (classifier layers) and replace the last max pooling layer by an aggregation pooling layer (more details about this layer in the next subsection)

dfeats = np.load(models['alexnet-cls-lm-gem']['dataset'])
qfeats = np.load(models['alexnet-cls-lm-gem']['queries'])
print(dfeats.shape)
# Q: Why does the size of the feature representation changes? Why does the size of the feature representation is important for a image retrieval task?
dataset.vis_top(dfeats, q_idx, q_feat=qfeats[qidx])

do_tsne(dfeats, labels, classes, sec='1c')
# Q: How does the aggregation layer changes the t-SNE visualization? Can we see some structure in the clusters of similarly labeled images?




#### 1d: Resnet18 architecture with GeM pooling ####
model_0 = resnet18()
model_1d = resnet18_GeM()
print(model_0.adpool)
print(model_1a.adpool)
# Q: Why do we change the average pooling layer of the original Resnet18 architecture for a generalized mean pooling? What operation is the layer model_1a.adpool doing?
# Hint: You can see the code of the generalized mean pooling in file pooling.py

# load oxford features from ResNet18 model
dfeats = np.load(models['resnet18-cls-lm-gem']['dataset'])
qfeats = np.load(models['resnet18-cls-lm-gem']['queries'])
print(norm(dfeats[:10], axis=1))
print(dfeats.shape)
# visualize top results for a given query index
q_idx = 0
dataset.vis_top(dfeats, q_idx, q_feat=qfeats[qidx])

do_tsne(dfeats, labels, classes, sec='1d')
# Q: How does this model compare with model 1c, that was trained in the same dataset for the same task? How does is compare to the finetuned models of 1b?




#### 1e: PCA Whitening ####

# We use a PCA learnt on landmarks to whiten the output features of 'resnet18-cls-lm-gem'
dfeats = np.load(models['resnet18-cls-lm-gem-pcaw']['dataset'])
qfeats = np.load(models['resnet18-cls-lm-gem-pcaw']['queries'])
do_tsne(dfeats, labels, classes, sec='1e')
# run t-SNE including unlabeled images
do_tsne(dfeats, labels, classes, sec='1e', show_unlabeled=True)
# Q: What can we say about the separation of data when included unlabeled images? And the distribution of the unlabeled features? How can we train a model to separate labeled from unlabeled data?




#### 1f: Finetuning on Landmarks for retrieval ####
# Now we learn the architecture presented in 1e in and end-to-end manner for the retrieval task
# The architecture includes a FC that replaces the PCA projection
dataset.vis_triplets()

dfeats = np.load(models['resnet18-rnk-lm-gem']['dataset'])
qfeats = np.load(models['resnet18-rnk-lm-gem']['queries'])
dataset.vis_top(dfeats, q_idx, q_feat=qfeats[qidx])
do_tsne(dfeats, labels, classes, sec='1f')
do_tsne(dfeats, labels, classes, sec='1f', show_unlabeled=True)
# Q: Compare the plots with unlabeled data of the model trained for retrieval (with triplet loss) and the model trained for classification of the previous subsection. How does it change?




#### 1g: Data augmentation ####
# This model has been trained the following data augmentation:
# cropping, pixel jittering, rotation, tilting
dfeats = np.load(models['resnet18-rnk-lm-gem-da']['dataset'])
qfeats = np.load(models['resnet18-rnk-lm-gem-da']['queries'])
dataset.vis_top(dfeats, q_idx, q_feat=qfeats[qidx])

do_tsne(dfeats, labels, classes, sec='1g')




#### 1h: Multi-resolution ####
dfeats = np.load(models['resnet18-rnk-lm-gem-da-mr']['dataset'])
qfeats = np.load(models['resnet18-rnk-lm-gem-da-mr']['queries'])
dataset.vis_top(dfeats, q_idx, q_feat=qfeats[qidx])

do_tsne(dfeats, labels, classes, sec='1g')




# 1i: Improved architectures

#######################################
####      Section 2: Testing       ####
#######################################

q_idx = 0
feats = np.load('data/features/resnet50-rnk-lm-da_ox.npy')

# load weights:
model = resnet50_rmac()

model.eval()
# Q: What does it change in the model's architecture when we pass it to evaluation mode?
# Hint: Which layers used in training are not useful for testing?

# evaluate model for query
q_feat = q_eval(model, dataset, q_idx)
dataset.vis_top(feats, q_idx, q_feat)


#### Section 2a: Robustness to input transformations

q_feat = q_eval(model, dataset, q_idx, flip=True)
dataset.vis_top(feats, q_idx, q_feat)
# Q1: What is the impact of flipping the query image?

q_feat = q_eval(model, dataset, q_idx, rotate=5.)
dataset.vis_top(feats, q_idx, q_feat)
# Q2: Change the rotation value (in +/- degrees). What is the impact of rotating it? Up to which degree of rotation is the result stable?


#### Section 2b: Queries with multi-scale features

q_feat = q_eval(model, dataset, q_idx, scales=1)
dataset.vis_top(feats, q_idx, q_feat)

q_feat = q_eval(model, dataset, q_idx, scales=2)
dataset.vis_top(feats, q_idx, q_feat)
# Q: What is the impact of using more scales?


#### Section 2c: Robustness to resolution changes

q_feat = q_eval(model, dataset, q_idx, resize=1.5)
dataset.vis_top(feats, q_idx, q_feat)
# Q: Resize the image by a factor. What is the impact of resizing it, especially to very low resolution?


#### Subsection 2d: Robustness to compression (using PQ)

m = 256      # number of subquantizers
n_bits = 8   # bits allocated per subquantizer

feats_train = np.load('data/features/resnet50-rnk-lm-da_ox.npy')
dataset.pq_train(feats_train, m, n_bits)

# dataset to encode
dataset.pq_add(feats)

# search:
dataset.vis_top(feats, q_idx, pq_flag=True)
# Q1: How much memory (in bytes) is needed to store the compressed representation?
# Q2: What is the compression ratio?
# Q3: How did the compression affect the retrieval results?
# Q4: Change the values and m & n_bit and observe the change in retrieval performance.


#### Subsection 2e: Average query expansion

dataset.vis_top(feats, q_idx, nqe=3)
# nqe is the number of database items with which to expand the query.
# Q1: What is the impact of using different values of nqe?

## Subsection 2f: alpha query expansion

dataset.vis_top(feats, q_idx, nqe=5, aqe=3.0)
# aqe is the value of alpha applied for alpha query expansion.
# Q1: How should nqe be chosen? Hint: What is the impact of low prec@K (where K is equivalent to nqe) on aqe?
# Q2: What is the impact of using different values of nqe, aqe?


#### Subsection 2g: Diffusion

q_idx = 0
dataset.vis_top(feats, q_idx)
dataset.vis_top(feats, q_idx, dfs='it:int20')
# Parameters for dfs are passed as strings with datatypes indicated. The default parameter string is:
#    'alpha:float0.99_it:int20_tol:float1e-6_gamma:float3_ks:100-30_trunc:bool_bsize:int100000_fsr:bool_IS:bool_wgt:bool_bs:bool_reg:bool_split:int0_gmp:bool'
#    strings passed to the dfs parameter overwrite the default parameters

# Q1: The affinity matrix is computed using the similarity measure s = <f_i, f_j>^alpha, where 0 < alpha <= 1.0. Use dfs='alpha:float<alpha>' for different values of alpha. What is the impact of changing it? E.g:
dataset.vis_top(feats, q_idx, dfs='alpha:float0.8')

# Q2: k_q is the number of database items to use for diffusion. Use dfs='ks:100-<k_q>' for different values of k_q. What is the impact of changing it? E.g:
dataset.vis_top(feats, q_idx, dfs='ks:100-5')

# Q3: trunc is the number of sub-rows and columns to use for diffusion. Use dfs='trunc:int<trunc>' for different values of trunc. What is the impact of changing it? E.g:
dataset.vis_top(feats, q_idx, dfs='trunc:int2000')
# Q4: What is the maximum value of trunc and what case does it generalize to?
