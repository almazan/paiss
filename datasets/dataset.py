import os
import json
import pdb
import numpy as np


class Dataset(object):
    ''' Base class for a dataset. To be overloaded.
    Contains:
        - images                --> get_filename(i)
        - image labels          --> get_label(i)
        - list of image queries --> get_query_filename(i)
        - list of query ROIs    --> get_query_roi(i)

    Creation:
        Use dataset.create( "..." ) to instanciate one.
        db = dataset.create( "ImageList('path/to/list.txt')" )

    Attributes:
        root:       image directory root
        nimg:       number of images == len(self)
        nclass:     number of classes
        nquery:     number of queries
    '''
    root = ''
    nimg = 0
    nclass = 0
    ninstance = 0
    nquery = 0

    labels = [] # all labels
    c_relevant_idx = {} # images belonging to each class
    #c_non_relevant_idx = {} # images not belonging to each class

    def __len__(self):
        return self.nimg

    def get_key(self, i):
        raise NotImplementedError()

    def get_filename(self, i):
        return os.path.join(self.root, self.get_key(i))

    def get_image(self, i, roi=None):
        from PIL import Image
        img = Image.open(self.get_filename(i)).convert('RGB')
        if roi is not None:
            img = img.crop(tuple(roi))
        return img

    def get_label(self, i, toint=False):
        raise NotImplementedError()

    def has_label(self):
        try: self.get_label(0); return True
        except NotImplementedError: return False
    
    def class_name(self, i):
        raise NotImplementedError()

    def get_query_filename(self, i):
        raise NotImplementedError()

    def get_query_roi(self, i):
        raise NotImplementedError()

    def original(self):
        # overload when the dataset is derived from another
        return self

    def __repr__(self):
        res =  'Dataset: %s\n' % self.__class__.__name__
        res += '  %d images' % len(self)
        if self.nclass: res += ", %d classes" % (self.nclass)
        if self.ninstance: res += ', %d instances' % (self.ninstance)
        if self.nquery: res += ', %d queries' % (self.nquery)
        res += '\n  root: %s...\n' % self.root
        return res



























