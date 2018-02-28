import os
from datasets.imdb import imdb
import numpy as np
import cPickle
from fast_rcnn.config import cfg
import scipy

class caltech(imdb):
    def __init__(self, image_set='train', data_path='/home/yxchen/rpn_bf/datasets/caltech_reduce1'):
        imdb.__init__(self, 'caltech_' + image_set)

        self._data_path = os.path.join(data_path, image_set)

        self._imdb_txt = os.path.join(self._data_path, 'imdb.txt')
        self._roidb_txt = os.path.join(self._data_path, 'roidb.txt')

        self._image_set = image_set

        self._image_format = '.jpg'

        self._classes = ('__background__', # always index 0
                         'pedestrian')

        self._image_files, self._gt_roidb = self._load_imdb_roidb()
        self._image_index = self._image_files

    def image_path_at(self, i):
        image_path = os.path.join(self._data_path, 'images', self._image_files[i] + self._image_format)
        return image_path
    
    def _load_imdb_roidb(self):
        with open(self._imdb_txt, 'r') as f:
            _list = f.readlines()
            _list = [img.strip() for img in _list]

        gt_roidb = []

        print 'load roidb from: ', self._roidb_txt
        print 'classes: ', self.num_classes

        rois = [[] for i in range(len(_list))]
        with open(self._roidb_txt, 'r') as f:
            for line in f.readlines():
                line_split = line.split(' ')

                idx = int(line_split[0])
                x1 = float(line_split[1])
                y1 = float(line_split[2])
                x2 = float(line_split[3])
                y2 = float(line_split[4])
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                #x1 = 1 if x1 < 1 else x1
                #x2 = 639 if x2 > 639 else x2
                ignore = int(line_split[5])

                if ignore != 1:
                    rois[idx - 1].append({'box': [x1, y1, x2, y2],
                                'ignore': ignore,
                                'gt_class': 1,
                                'overlap': 1,
                                'seg_area': (x2 - x1 + 1) * (y2 - y1 + 1)})

        for i in range(len(rois)):
            num_rois = len(rois[i])
            boxes = np.zeros((num_rois, 4), dtype=np.int16)
            gt_classes = np.zeros((num_rois), dtype=np.int32)
            overlaps = np.zeros((num_rois, self.num_classes), dtype=np.float32)
            seg_areas = np.zeros((num_rois), dtype=np.float32)
                    
            for j in range(len(rois[i])):
                roi = rois[i][j]
                
                boxes[j, :] = roi['box']
                gt_classes[j] = roi['gt_class']
                overlaps[j, 1] = roi['overlap']
                seg_areas[j] = roi['seg_area']

            overlaps = scipy.sparse.csr_matrix(overlaps)

            gt_roidb.append({'boxes': boxes,
                           'gt_classes': gt_classes,
                           'gt_overlaps': overlaps,
                           'flipped': False,
                           'seg_areas': seg_areas})

        ## because py-faster-rcnn does this for us, we do not need to do it again
        #cnt = 0
        #for i in range(len(rois) - 1, -1, -1):
        #    #print len(rois[i])
        #    if len(rois[i]) == 0:
        #        del gt_roidb[i]
        #        del _list[i]
        #    else:
        #        cnt += 1
        #print 'total: ', cnt
        #print '464: ', _list[464], gt_roidb[464]

        return _list, gt_roidb

    def gt_roidb(self):
        return self._gt_roidb

    def rpn_roidb(self):
        gt_roidb = self.gt_roidb()
        rpn_roidb = self._load_rpn_roidb(gt_roidb)
        roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)

        return roidb

