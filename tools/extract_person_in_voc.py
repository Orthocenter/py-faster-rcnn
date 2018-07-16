import _init_paths
from datasets.pascal_voc import pascal_voc
import os
from shutil import copyfile
import cv2

voc = pascal_voc('trainval', '2007')
roidb = voc.gt_roidb()

for i in xrange(len(roidb)):
    path = voc.image_path_at(i)

    img = cv2.imread(path)
    width = img.shape[1]
    height = img.shape[0]
    fw = 640. / width
    fh = 480. / height
    img = cv2.resize(img, (640, 480))
    cv2.imwrite('/home/yxchen/voc2007_rpn/images/' + path[75:], img)

    annot_name = path[75:-4] + '.txt'

    with open('/home/yxchen/voc2007_rpn/annotations/' + annot_name, 'w') as f:
        f.write("% bbGt version=3\n")
        for j in xrange(len(roidb[i]['gt_classes'])):
            if roidb[i]['gt_classes'][j] == 15:
                box = roidb[i]['boxes'][j]
                box[0] = box[0] * fw
                box[1] = box[1] * fh
                box[2] = box[2] * fw
                box[3] = box[3] * fh
                f.write("person {} {} {} {} 0 0 0 0 0 0 0\n".format(box[0], box[1], box[2] - box[0], box[3] - box[1]))
    
