#!/usr/bin/env python
# --------------------------------------------------------
# Fast/er/ R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Generate RPN proposals."""

import _init_paths
import numpy as np
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from rpn.generate import im_proposals
import cPickle
import caffe
import argparse
import pprint
import time, os, sys
from utils.timer import Timer
import matplotlib.pyplot as plt
import cv2

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=1, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default="/home/yxchen/py-faster-rcnn-original/models/pascal_voc/VGG16/faster_rcnn_alt_opt/rpn_test.pt", type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default="/home/yxchen/py-faster-rcnn-original/data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel", type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    #if len(sys.argv) == 1:
    #    parser.print_help()
    #    sys.exit(1)

    args = parser.parse_args()
    return args

def _vis_proposals(im, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    class_name = 'obj'
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w > 150 or h > 150:
            continue
        ap = float(w) / h
        if ap > 0.6 or ap < 0.3:
            continue

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    # RPN test settings
    cfg.TEST.RPN_PRE_NMS_TOP_N = -1
    cfg.TEST.RPN_POST_NMS_TOP_N = 2000
    cfg.TEST.RPN_NMS_THRESH = 0.7

    print('Using config:')
    pprint.pprint(cfg)

    while not os.path.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    import glob
    imdb = glob.glob("/home/yxchen/RPN_BF/datasets/caltech/train/images/*")[35900 + 3700 + 800:]

    _t = Timer()
    imdb_boxes = [[] for _ in xrange(len(imdb))]
    for i in xrange(len(imdb)):
        with open("proposals/" + imdb[i][-21:] + ".txt", "w") as f:
            im = cv2.imread(imdb[i])
            _t.tic()
            imdb_boxes[i], scores = im_proposals(net, im)
            _t.toc()
            print 'im_proposals: {:d}/{:d} {:.3f}s' \
                    .format(i + 1, len(imdb), _t.average_time)

            for j in xrange(len(imdb_boxes[i])):
                box = imdb_boxes[i][j]
                score = scores[j]

                f.write("%d %d %d %d " % (box[0], box[1], box[2], box[3]))
                f.write("%f\n" % score)

            if 0:
                dets = np.hstack((imdb_boxes[i], scores))
                # from IPython import embed; embed()
                _vis_proposals(im, dets[:60, :], thresh=0.6)
                plt.show()

    #output_dir = get_output_dir(imdb, net)
    #rpn_file = os.path.join(output_dir, net.name + '_rpn_proposals.pkl')
    #with open(rpn_file, 'wb') as f:
    #    cPickle.dump(imdb_boxes, f, cPickle.HIGHEST_PROTOCOL)
    #print 'Wrote RPN proposals to {}'.format(rpn_file)
