from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cmx

np.random.seed(0)

def do_tsne(feats, labs, cls, show_unlabeled=False, sec=''):
    # hyperparameters:
    n_pca  = 100
    n_sne  = 1000
    n_cls = 12
    n_dim  = 2
    n_iter = 300

    # prepare data:
    labs  = [int(i) for i in labs]
    slabs = list(set(labs))

    # sample classes:
    if not isinstance(cls, np.ndarray): cls = np.array(cls)
    cls     = np.array( ['unlabeled']+cls.tolist() )
    cls_ind = list(range(feats.shape[0]))#np.array([e for e,i in enumerate(labs) if i in set(cls)])
    featsc  = feats[cls_ind,:]
    labsc   = np.array([cls[labs[i]] for i in cls_ind])

    #reduce dimensionality:
    print('applying PCA...')
    pca   = PCA(n_components=n_pca)
    feats = pca.fit_transform(featsc)
    n_smp = feats.shape[0]

    print('applying t-SNE...')
    time_start = time.time()
    tsne = TSNE(n_components=n_dim, verbose=1, perplexity=40, n_iter=n_iter)
    feats_tsne = tsne.fit_transform(featsc)

    print('t-SNE done! Time elapsed: {:.3f} seconds'.format(time.time()-time_start))

    # Visualize the results
    fig, ax   = plt.subplots()
    values    = range(n_cls) if show_unlabeled else range(1,n_cls)
    jet  = cm = plt.get_cmap('jet') 
    cNorm     = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    for c in values:
        colorVal  = scalarMap.to_rgba(c)
        embed_idx = np.where(labsc == cls[c])[0] 
        embed_x   = feats_tsne[embed_idx,0]
        embed_y   = feats_tsne[embed_idx,1]
        ax.scatter(embed_x, embed_y, c=colorVal, label=cls[c])

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
    plt.axis('off')
    figname = 'scatter_niter{:03d}_'.format(n_iter)+sec
    if show_unlabeled: figname += 'showunlabeled' 
    figname += '.png'
    plt.show(block=False)
    plt.savefig(figname, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run t-SNE on features representations for retrieval')
    parser.add_argument('--dataset', '-d', type=str, default='oxford', choices=('oxford'), help='selects the dataset')
    parser.add_argument('--backbone', '-b', type=str, default='alexnet', choices=('alexnet', 'resnet18', 'resnet50'), help='selects the backbone architecture')
    parser.add_argument('--trained', '-t', type=str, default='classification', help='selects the task used to train the model')
    parser.add_argument('--finetuned', '-f', type=bool, default=False, help='switchs if the model is finetuned (trained on landmarks) or not (trained on Imagenet)')
    parser.add_argument('--show_unlabeled', '-s', type=bool, default=False, help='show unlabeled data')

    args = parser.parse_args()
    train_db = 'landmark' if args.finetuned else 'imagenet'
    data_f = 'data/'+args.dataset+'_res_224_'+args.backbone+'_'+train_db+'_'+args.trained+'.npz'
#    data_f = 'data/lm18_val_feats_labels.npz'
#    data_f = 'data/oxford_res_224_alexnet_imagenet_classification.npz'
    data   = np.load(data_f)
    feats  = data['feats']
    labs   = data['labs']
    cls = ['landmark 1','landmark 2','landmark 3','landmark 4','landmark 5','landmark 6','landmark 7','landmark 8','landmark 9','landmark 10','landmark 11']

    do_tsne(feats, labs, cls, sec='test')
