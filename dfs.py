import numpy as np
import scipy.sparse as ssparse
import scipy.sparse.linalg
import torch
import warnings

def get_knn(fa, fb, k, dim=1):
    sim     = torch.matmul(fa, torch.t(fb))
    ss, ind = torch.topk(sim, k, dim=dim)

    # clean up:
    del sim

    return ss, ind

def knn_graph(ss, ind):
    n, k = ss.size()

    # sparse matrix parameters:
    I    = (np.ogrid[:n,:0][0] * np.ones((1,k), dtype=np.float32)).flatten().astype(int)
    J    = np.hstack(ind)
    sim  = np.hstack(ss)
    sim[sim < 0] = 0

    # remove diagonal entries:
    mask = I != J
    I    = I[mask]
    J    = J[mask]
    sim  = sim[mask]

    # create affinity matrix:
    A = ssparse.coo_matrix((sim, (I, J)), shape=(n,n)).tocsr()

    # make graph mutual:
    A = A.multiply(A.multiply(A.T).power(0.))

    return A

def transition_matrix(W):
    n,m = W.shape
    with warnings.catch_warnings(): # suppress divide-by-zero warning
        warnings.simplefilter("ignore")
        D = W.dot(np.ones((m,1),dtype=np.float32)) ** -0.5
    D  = ssparse.diags(D.squeeze())
    S  = D.dot(W).dot(D)

    return S,D

def get_y(v, qv, k, gamma, wgt=False):
    """
     construction of vector y for the query vector
     y = get_y(V, qv, k, gamma)
     v:  dataset vectors
     qv: query vector
     k:  number of nearest neighors to keep
     gamma: similarity exponent  
    """

    if qv.dim() < 2:
        qv = qv[np.newaxis,]

    s, knn = get_knn(v, qv, k, dim=0)
    N      = v.shape[0]
    sc     = np.bincount(knn.numpy().flatten().astype(int),
                         s.numpy().flatten(),
                         N)
    s, knn = [np.sort(sc)[::-1], np.argsort(sc)[::-1]]
  
    # normalize (hacky):
    s /= float(qv.shape[0])

    # create y:
    y = np.zeros((N,1), dtype=np.float32)
    y[knn[0:k],0] = np.maximum(s[0:k] ** gamma, 0)

    return y

def dfs(A, y, tol=1e-10, it=100):
    # function to perform diffusion
    # solving system A*f = y

    f,_ = ssparse.linalg.cg(A,y,tol=tol,maxiter=it)

    return f

def parse_args(dargs=''):
    """
        alpha: alpha for diffusion
        it: iterations for CG
        tol: tolerance for CG
        gamma: similarity exponent
    """

    default_args = 'alpha:float0.99_it:int20_tol:float1e-6_gamma:float3_ks:100-30_trunc:bool_bsize:int100000_fsr:bool_IS:bool_wgt:bool_bs:bool_reg:bool_split:int0_gmp:bool'

    dargs = default_args + '_' + dargs if len(dargs) > 0 else default_args
    dargs = dargs.split('_')
    dargs = [a.split(':') for a in dargs]
    adict = {}
    for k,v in dargs:
        if 'int' in v:
            adict[k] = int(v.replace('int',''))
        elif 'float' in v:
            adict[k] = float(v.replace('float',''))
        elif 'bool' in v:
            adict[k] = bool(v.replace('bool',''))
        else:
            adict[k] = v

    return adict

def reg_diffusion(q_feat, d_feats, dargs='', out_dir='.'):
    q_feat  = torch.from_numpy(q_feat)
    d_feats = torch.from_numpy(d_feats)

    # parse args:
    dargs = parse_args(dargs)
    
    nd    = d_feats.size()[0]

    # Diffusion parameters
    kd, kq = [int(k) for k in dargs['ks'].split('-')]

    # Enable truncation?:
    if dargs['trunc']:
        do_trunc = True
        topn     = dargs['trunc']
    else:
        do_trunc = False

    # Create the graph
    print('Computing kNN graph...\n')
    ss, ind = get_knn(d_feats, d_feats, kd)
    A_      = knn_graph(torch.pow(ss, dargs['gamma']), ind)

    # in case of truncation the Laplacian is computed per query
    if not do_trunc:
        T, _ = transition_matrix(A_)
        A    = ssparse.eye(T.shape[0], dtype=np.float32) - dargs['alpha'] * T
        del A_, T

    # Query
    print('Performing diffusion...\n') 
    if not do_trunc:
        # construction of y vector
        y = get_y(d_feats, q_feat, kq, dargs['gamma'])

        # diffusion
        f = dfs(A,y,dargs['tol'],dargs['it']).T
    else:
        # sub-index for truncation
        sub = get_knn(d_feats, q_feat[np.newaxis,], topn, dim=0)[1].squeeze().tolist()

        # truncated Laplacian
        Asub = A_[sub,:].tocsc()[:,sub].tocsr()
        T, _ = transition_matrix(Asub)

        # construction of y vector
        subt = torch.LongTensor(sub)
        d_feats_sub = d_feats.index_select(0,subt)
        y = get_y(d_feats_sub, q_feat, kq, dargs['gamma'])
        f = -1e6 * np.ones((nd), dtype=np.float32)

        # diffusion 
        L      = ssparse.eye(T.shape[0]) - dargs['alpha'] * T
        f[sub] = dfs(L,y,dargs['tol'],dargs['it'])

    return f

if __name__ == "__main__":
    data_f = 'data/features/resnet50-rnk-lm-da_ox.npy'
    feats  = np.load(data_f)
    sim    = reg_diffusion(feats[0], feats)
