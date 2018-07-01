import os
import os.path as osp
import pdb
import json
import sys
from collections import OrderedDict
from sklearn.metrics import average_precision_score
import numpy as np
import subprocess
import functools
import getpass
import matplotlib.pyplot as plt

if os.name != 'nt': import faiss

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # OMP issue described here: https://github.com/dmlc/xgboost/issues/1715

from .dataset import Dataset
from dfs import reg_diffusion

class Oxford(Dataset):
    def __init__(self, dataset_name='Oxford'):
        assert dataset_name == 'Oxford'
        self.dataset_name = dataset_name
        self.__load()

    @property
    def nquery(self):
        return self.__N_queries

    @property
    def nclass(self):
        return len(self.__cls_names)

    @property
    def N_images(self):
        return self.__N_images

    def get_key(self, i):
        return self.__img_filenames[i]

    def get_label(self, i):
        return self.__lab[i]

    def get_query(self, i):
        return self.__q_index[i]

    def get_query_list(self):
        return self.__q_index

    def get_label_vector(self):
        return self.__lab

    def get_label_names(self):
        return self.__cls_names

    def get_relevants(self):
        return self.__relevants

    def get_filename(self, i):
        try:
            return os.path.normpath("{0}/{1}.{2}".format(self.__img_root, self.__img_filenames[i], self.__dataset_extensions[i]))
        except:
            return os.path.normpath("{0}/{1}.{2}".format(self.__img_root, self.__img_filenames[i], self.__dataset_extension))

    def get_query_filename(self, i):
        try:
            return os.path.normpath("{0}/{1}.{2}".format(self.__img_root, self.__img_filenames[self.__q_index[i]], self.__dataset_extensions[i]))
        except:
            return os.path.normpath("{0}/{1}.{2}".format(self.__img_root, self.__img_filenames[self.__q_index[i]], self.__dataset_extension))

    def get_query_roi(self, i):
        return self.__q_roi[self.__q_names[i]]

    def vis_top(self, feats, q_idx, q_feat=None, topk=15, nqe=0, aqe=0.0, pq_flag=False, dfs='', out_image_file=None, ap_flag=False, query_image=None):
        q_name = self.__q_names[q_idx]
        q_idx = self.__q_index[q_idx]
        if q_feat is None: q_feat = feats[q_idx]

        if pq_flag:
            # perform AQE?
            if nqe > 0:
                _, idx = self.pq_search(q_feat, k=nqe)
                idx    = idx[0]
                q_aug  = np.vstack([feats[j] * np.exp(-D[j])**aqe for j in idx])
                q_aug  = np.mean(np.vstack((q_feat, q_aug)), axis=0)
                q_aug  = q_aug / np.linalg.norm(q_aug)
                _, idx = self.pq_search(q_aug, k=self.N_images)
            else:
                _, idx = self.pq_search(q_feat, k=self.N_images)
            idx = idx[0]
        else:
            sim = np.dot(q_feat, feats.T)

            # perform AQE?
            if nqe > 0:
                idx   = np.argpartition(sim, -nqe)[-nqe:]
                q_aug = np.vstack([feats[j] * sim[j]**aqe for j in idx])
                q_aug = np.mean(np.vstack((q_feat, q_aug)), axis=0)
                q_f   = q_aug / np.linalg.norm(q_aug)
            else:
                q_f   = q_feat

            # perform diffusion?
            if dfs:
                sim = reg_diffusion(q_f, feats, dfs)
            else:
                sim = np.dot(q_f, feats.T)

            idx = np.argsort(sim)[::-1]

        # compute ap?
        if ap_flag:
            ap = self.get_ap(q_name, idx)
            print('AP={:.2f}'.format(ap * 100.))

        # visualize:
        nplots = 1 + topk
        bsize  = 20
        qlab   = self.__lab[q_idx]
        plt.ion()
        fig, axes = plt.subplots(1, nplots, figsize=(20, 3))
        if query_image is None:
            query_image = self.get_image(q_idx, self.__q_roi[q_name])
        axes[0].imshow(self.draw_border(np.array(query_image),bsize,[1,1,1]))
        axes[0].set_axis_off()
        axes[0].set_title('query')
        for i in range(1,nplots):
            if idx[i-1] in self.__relevants[q_name]:
                bcol = [0,1,0]
            elif idx[i-1] in self.__non_relevants[q_name]:
                bcol = [1,0,0]
            else:
                bcol = [0.5, 0.5, 0.5]
            axes[i].imshow(self.draw_border(np.array(self.get_image(idx[i - 1])),bsize,bcol))
            axes[i].set_axis_off()
            axes[i].set_title("")

        plt.subplots_adjust(top=0.91, bottom=0.01, left=0.01, right=0.99)

        # save/show fig:
        plt.show()
        input("Showing the top-{} results. Press Enter to continue...".format(topk))
        if out_image_file is not None:
            if not os.path.exists(os.path.dirname(out_image_file)):
                os.makedirs(os.path.dirname(out_image_file))
            plt.savefig(out_image_file)
        plt.close()

    def vis_queries(self):
        for i, q_idx in enumerate(self.__q_index):
            assert self.__lab[q_idx] == (1+i//5)
            if i%5 == 0:
                fig = plt.subplots()
                plt.subplots_adjust(0,0,1,0.95)
            cls = self.__cls_names[self.__lab[q_idx]-1]
            plt.subplot(3,2,i%5+1)
            plt.imshow(self.get_image(q_idx))
            plt.axis('off')
            if i%5 == 0: plt.title(cls)
            if i%5 == 4:
                plt.show(block=False)
                input('Queries for landmark %s. Press enter to continue' % cls)
                plt.close()
        return

    def vis_triplets(self):# untested
        max_plot = 6
        q_idxs = np.random.choice(range(54), max_plot)
        q_label = q_idxs//5+1
        p_idxs = np.random.choice(range(5), max_plot)
        n_idxs = np.random.choice(range(1000), max_plot)
        for i in range(max_plot):
            fig, axes = plt.subplots(1,3,figsize=(20,3))
            axes[0].imshow(self.get_image(self.__q_index[q_idxs[i]]))
            axes[0].set_title("query")
            axes[0].set_axis_off()

            axes[1].imshow(self.draw_border(np.array(self.get_image( np.nonzero(self.__lab == q_label[i])[0][p_idxs[i]] )),30,[0,1,0]))
            axes[1].set_title('positive ex')
            axes[1].set_axis_off()

            axes[2].imshow(self.draw_border(np.array(self.get_image( np.nonzero(self.__lab != q_label[i])[0][n_idxs[i]] )),30,[1,0,0]))
            axes[2].set_title('negative ex')
            axes[2].set_axis_off()
            plt.show(block=False)
            input('Triplet for landmark %s. Press enter to continue' % self.__cls_names[q_label[i]-1])
            plt.close()
        return

    def pq_search(self, x_q, k=20):
        if x_q.ndim == 1:
            x_q = np.ascontiguousarray(x_q[np.newaxis])

        # Perform a search:
        D, I = self.pq.search(x_q, k)

        return D, I

    def pq_add(self, x_d):
        # Populate the database:
        self.pq.add(x_d)

    def pq_train(self, x, m, n_bits):
        d  = x.shape[1]

        # Create the index
        self.pq = faiss.IndexPQ(d, m, n_bits)

        # Training
        self.pq.train(x)

    def get_ap(self, q_name, sorted_idx):
        rel   = self.__relevants[q_name]
        junk  = self.__junk[q_name]

        # construct ground-truth and scores:
        y_scores = np.zeros(self.N_images)
        y_true   = np.zeros(self.N_images)
        for e,i in enumerate(sorted_idx): y_scores[i] = self.N_images - e
        for i in rel: y_true[i] = 1

        # remove junk:
        y_scores = np.delete(y_scores, junk)
        y_true   = np.delete(y_true, junk)

        # compute ap:
        return average_precision_score(y_true, y_scores)

    def __load(self):
        # Check localtion and deploy if needed:
        root = self.__check_location()
        self.root = root
        # Prepare image list and labels
        self.__dataset_extension = "jpg"
        self.__lab_root = '{0}/lab/'.format(root)
        self.__img_root = '{0}/jpg/'.format(root)
        lab_filenames = np.sort(os.listdir(self.__lab_root))
        self.__img_filenames = [e[:-(len(self.__dataset_extension) + 1)] for e in np.sort(os.listdir(self.__img_root))]
        self.nimg = len(self.__img_filenames)
        # Parse the label files to
        # i) map names to filenames and vice versa
        # ii) get the relevant regions of the queries,
        # iii) get the indexes of the dataset images that are queries
        self.__relevants = {}
        self.__junk = {}
        self.__non_relevants = {}

        self.__filename_to_name = {}
        self.__name_to_filename = OrderedDict()
        self.__q_roi = {}
        for e in lab_filenames:
            if e.endswith('_query.txt'):
                q_name = e[:-len('_query.txt')]
                q_data = open("{0}/{1}".format(self.__lab_root, e)).readline().split(" ")
                q_filename = q_data[0][5:]  # Remove the oxc1_ prefix
                self.__filename_to_name[q_filename] = q_name
                self.__name_to_filename[q_name] = q_filename
                good = set([e.strip() for e in open("{0}/{1}_ok.txt".format(self.__lab_root, q_name))])
                good = good.union(set([e.strip() for e in open("{0}/{1}_good.txt".format(self.__lab_root, q_name))]))
                junk = set([e.strip() for e in open("{0}/{1}_junk.txt".format(self.__lab_root, q_name))])
                good_plus_junk = good.union(junk)
                self.__relevants[q_name] = [i for i in range(len(self.__img_filenames)) if self.__img_filenames[i] in good]
                self.__junk[q_name] = [i for i in range(len(self.__img_filenames)) if self.__img_filenames[i] in junk]
                self.__non_relevants[q_name] = [i for i in range(len(self.__img_filenames)) if self.__img_filenames[i] not in good_plus_junk]
                self.__q_roi[q_name] = np.array(list(map(float, q_data[1:])), dtype=np.float32)

        self.__q_names = list(self.__name_to_filename.keys())
        self.__q_index = np.array([self.__img_filenames.index(self.__name_to_filename[qn]) for qn in self.__q_names])
        self.__N_images = len(self.__img_filenames)
        self.__N_queries = len(self.__q_index)

        self.__find_labels()

    def __find_labels(self):
        self.__cls_names = np.unique([q_name[:-2] for q_name in self.__relevants.keys()])
        self.__lab = np.zeros(self.__N_images, dtype=np.int)
        for cls_idx, cls in enumerate(self.__cls_names):
            self.__lab[self.__relevants[cls+'_1']] = cls_idx+1

    def __check_location(self):
        root_ram = "."
        path = osp.join(root_ram, 'data/oxford5k')
        if not os.path.exists(path):
            raise NotImplementedError("Check your 'data/oxford5k' folder. If it is empty, follow the instructions in the README to download the dataset.")
        return path

    def draw_border(self,img,bsize,bcol):
        idim  = img.shape
        m     = idim[0]
        n     = idim[1]
        p     = 3
        maxv  = max(img.flatten())
        if maxv > 1:
            maxv = 255.0
        else:
            maxv = 255.0
            img  = img * maxv

        if len(idim) == 2:
            img = np.tile(img[:,:,np.newaxis],[1,1,p])

        new_img = maxv * np.ones((m+2*bsize,n+2*bsize,p))
        for i in range(p):
            new_img[:,:,i] = bcol[i] * new_img[:,:,i]

        new_img[bsize:bsize+m,bsize:bsize+n,:] = img
        new_img = new_img.astype(np.uint8)

        return new_img


