import glob
import os
import argparse

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--input_dir', dest='input_dir',
                        help='input_dir',
                        default=None, type=str)
    parser.add_argument('--output_dir', dest='output_dir',
                        help='output_dir',
                        default=None, type=str)

    args = parser.parse_args()
    return args

args = parse_args()

proposal_path = "/home/yxchen/py-faster-rcnn-original/proposals"
output_path = "/home/yxchen/py-faster-rcnn-original/res/faster-rcnn-caltech-5k"

res = {}

for filename in glob.glob(proposal_path + "/*.txt"):
    with open(filename, "r") as f:
        for line in f.readlines():
            line = line.split(" ")
            x0, y0, x1, y1 = [int(line[i]) for i in range(4)]
            score = float(line[-1])
            w = x1 - x0
            h = y1 - y0

            setid = filename[-25:-20]
            vid = filename[-19:-15]
            fid = int(filename[-13:-8]) + 1

            set_res = res.get(setid, {})
            v_res = set_res.get(vid, {})
            tmp = v_res.get(fid, "")
            v_res[fid] = tmp + "%d,%f,%f,%f,%f,%f\n" % (fid, x0, y0, w, h, score)
            set_res[vid] = v_res
            res[setid] = set_res

for setid in sorted(res.keys()):
    output_dir = output_path + "/" + setid
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print output_dir
    for vid in sorted(res[setid].keys()):
        with open(output_dir + "/" + vid + ".txt", "w") as f:
            for fid in sorted(res[setid][vid].keys()):
                f.write(res[setid][vid][fid])
