#!/bin/bash

EXPR_NAME='REDUCE_010'
MAX_ITERS=16000
GPU=0

OUTPUT_DIR=/home/yxchen/py-faster-rcnn/output/caltech_train/$EXPR_NAME
DATASET_PATH=/home/yxchen/RPN_BF/datasets/caltech_$EXPR_NAME
mkdir -p $OUTPUT_DIR

python tools/train_rpn_caltech.py \
    --net_name=VGG16 \
    --weights=/home/yxchen/py-faster-rcnn/data/imagenet_models/vgg16.caffemodel \
    --dataset_path=$DATASET_PATH \
    --output_dir=$OUTPUT_DIR \
    --max_iters=$MAX_ITERS \
    --gpu=$GPU  |& tee $OUTPUT_DIR/train.log


NET_FILE=$OUTPUT_DIR/vgg16_rpn_stage1_iter_$MAX_ITERS.caffemodel
OUTPUT_DIR=/home/yxchen/py-faster-rcnn/output/caltech_proposals/$EXPR_NAME
mkdir -p $OUTPUT_DIR
python tools/caltech_proposal_generate.py \
     --net=$NET_FILE \
    --output_dir=$OUTPUT_DIR \
    --gpu=$GPU |& tee $OUTPUT_DIR/proposal.log

INPUT_DIR=$OUTPUT_DIR
OUTPUT_DIR=/home/yxchen/RPN_BF/external/code3.2.1/data-USA/res/$EXPR_NAME
mkdir -p $OUTPUT_DIR
python tools/caltech_proposal_eval.py \
    --input_dir=$INPUT_DIR \
    --output_dir=$OUTPUT_DIR |& tee $OUTPUT_DIR/eval.log
