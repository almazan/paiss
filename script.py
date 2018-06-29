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

from datasets import create
from archs import *
#from utils.test import q_eval TODO: uncomment this when test.py is commited
from utils.tsne import do_tsne

np.random.seed(0)

## Section 1: Training

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

# Subsection 1a: AlexNet architecture
model_1a = alexnet_imagenet()
print(model_1a)
# Q: The original AlexNet model is trained for classification on ImageNet dataset, that contains 1000 image classes. What is the number of dimensions of the output of the model? What does each dimension represents?

feats = np.load('data/features/alexnet-cls-imagenet-fc7_ox.npy')
print(norm(feats[:10], axis=1))
print(feats.shape)
# Q: What does each line of the matrix feats represent? Where does the dimension of these lines comes from and how do we extract these features?
# Hint: uncomment and run the following command
# model_1a_test = alexnet_imagenet_test(); print(model_1a_test)

# visualize top results for a given query
q_idx = 0 # TODO: find good query candidates
dataset.vis_top(feats, q_idx)

# run t-SNE
do_tsne(feats, labels, classes, sec='1a')
# Q: What can be observe from the t-SNE visualization? Which classes 'cluster' well? Which do not?

# Subsection 1b: Finetuning on Landmarks
model_1b = alexnet_lm()
print(model_1b)
# Q: Why do we change the last layer of the AlexNet architecture? How do we initialize the layers of model_1b for finetuning? 

feats = np.load('data/features/alexnet-cls-lm-fc7_ox.npy')
dataset.vis_top(feats, q_idx)
# run t-SNE
do_tsne(feats, labels, classes, sec='1b')
# Q: How does the visualization change after finetuning? What about the top results?

# question on how the architecture demands the resize of the input images (specifically, the fully connected layers) ##########

# Subsection 1c: Generalized Mean Pooling (GeM)
model_1c = alexnet_GeM()
print(model_1c)
# Q: For this model, we remove all fully connected layers (classifier layers) and replace the last max pooling layer by an aggregation pooling layer (more details about this layer in the next subsection)

feats = np.load('data/features/alexnet-cls-lm_ox.npy')
print(feats.shape)
# Q: Why does the size of the feature representation changes? Why does the size of the feature representation is important for a image retrieval task?
dataset.vis_top(feats, q_idx)

do_tsne(feats, labels, classes, sec='1c')
# Q: How does the aggregation layer changes the t-SNE visualization? Can we see some structure in the clusters of similarly labeled images?
del model_1a, model_1b, model_1c

# Subsection 1d: Resnet18 architecture with aggregation pooling
model_0 = resnet18()
model_1d = resnet18_GeM()
print(model_0.adpool)
print(model_1a.adpool)
# Q: Why do we change the average pooling layer of the original Resnet18 architecture for a generalized mean pooling? What operation is the layer model_1a.adpool doing?
# Hint: You can see the code of the generalized mean pooling in file pooling.py

# load oxford features from ResNet18 model
feats = np.load('data/features/resnet18-cls-lm_ox.npy')
print(norm(feats[:10], axis=1))
print(feats.shape)
# visualize top results for a given query index
q_idx = 0
dataset.vis_top(feats, q_idx)

do_tsne(feats, labels, classes, sec='1d')
# Q: How does this model compare with model 1c, that was trained in the same dataset for the same task? How does is compare to the finetuned models of 1b?

# Subsection 1e: PCA

# We learn PCA mean and standard deviation on landmarks and apply it to the output of model_1d
feats = np.load('data/features/resnet18-cls-lm-pca_ox.npy')
do_tsne(feats, labels, classes, sec='1e')
# run t-SNE including unlabeled images
do_tsne(feats, labels, classes, sec='1e', show_unlabeled=True)
# Q: What can we say about the separation of data when included unlabeled images? And the distribution of the unlabeled features? How can we train a model to separate labeled from unlabeled data? 

del model_1d

# Subsection 1f: Finetuning on Landmarks for retrieval
# For sections 1f to 1h, we use the architecture presented by model_1d
dataset.vis_triplets()

feats = np.load('data/features/resnet18-rnk-lm_ox.npy')
dataset.vis_top(feats, q_idx)
do_tsne(feats, labels, classes, sec='1f')
do_tsne(feats, labels, classes, sec='1f', show_unlabeled=True)
# Q: Compare the plots with unlabeled data of the model trained for retrieval (with triplet loss) and the model trained for classification of the previous subsection. How does it change?

# 1g: Data augmentation
feats = np.load('data/features/resnet18-rnk-lm-da_ox.npy')
dataset.vis_top(feats, q_idx) 

do_tsne(feats, labels, classes, sec='1g')

# 1h: Multi-resolution
feats = np.load('data/features/resnet18-rnk-lm-da_mr_ox.npy')
dataset.vis_top(feats, q_idx) 

do_tsne(feats, labels, classes, sec='1g')

# 1i: Improved architectures


# Subsection 2c: PQ compression
m = 256        # number of subquantizers
n_bits = 8   # bits allocated per subquantizer

q_idx = 0 
feats = np.load('/nfs/team/cv/PAISS/PAISS2018/data/features/oxford_resnet50-ranking-lm_crop-tilt-rot-pjit.npy')
feats_train = np.load('/tmp-network/user/jalmazan/l2/retrieval/scores_ranking/Oxford/resnet50/Landmarks2_clsLM18_S800_gem_adam_lr5_crop1_tilt15_rotate20_pixjit_10240/features_for_pca.npy')
dataset.pq_train(feats_train, m, n_bits)

# dataset to encode
dataset.pq_add(feats)

# search:
dataset.vis_top(feats, q_idx, pq_flag=True) 
# Q1: How much memory (in bytes) is needed to store the compressed representation?
# Q2: What is the compression ratio?
# Q3: How did the compression affect the retrieval results?
# Q4: Change the values and m & n_bit and observe the change in retrieval performance.

## Section 2: Testing
q_idx = 0
model1 = resnet18_rmac()
model2 = resnet101_rmac()

model1.eval()
model2.eval()
# Q: What does it change in the models' architecture when we pass it to evaluation mode? Hint: Which layers used in training are not useful for testing?

# evaluate model for query
q_feat1 = q_eval(model1, dataset, q_idx)
dataset.vis_top(feats, q_idx, q_feat1)

q_feat2 = q_eval(model2, dataset, q_idx)
dataset.vis_top(feats, q_idx, q_feat2)
## Subsection 2a: cropping, flipping, rotating
q_feat = q_eval(model, dataset, q_idx, flip=True)
dataset.vis_top(feats, q_idx, q_feat)

## Section 3: Re-ranking

## Subsection 3a: Query expansion

#AQE
dataset.vis_top(feats, q_idx, nqe=3)

#alphaQE
dataset.vis_top(feats, q_idx, nqe=5, aqe=3.0)
# Q: How should aqe be chosen? Hint: What is the impact of low prec@K on aqe=K?

# Q: What is the impact of using different values of K, alpha?

# Subsection 3b: Diffusion
# Q: What is the impact of truncation, using different values of k_q?
