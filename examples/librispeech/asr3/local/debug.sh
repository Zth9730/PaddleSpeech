#!/bin/bash

if [ $# -lt 2 ] && [ $# -gt 3 ];then
    echo "usage: CUDA_VISIBLE_DEVICES=0 ${0} config_path ckpt_name ips(optional)"
    exit -1
fi

ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
echo "using $ngpu gpus..."

config_path=$1
ckpt_name=$2
ips=$3

if [ ! $ips ];then
  ips_config=
else
  ips_config="--ips="${ips}
fi

mkdir -p exp

# seed may break model convergence
seed=1998
if [ ${seed} != 0 ]; then
    export FLAGS_cudnn_deterministic=True
fi

# export FLAGS_cudnn_exhaustive_search=true
# export FLAGS_conv_workspace_size_limit=4000

python3 -u /home/zhangtianhao/workspace/PaddleSpeech/paddlespeech/s2t/models/wav2vec2/wav2vec2_ASR.py

if [ $? -ne 0 ]; then
    echo "Failed in training!"
    exit 1
fi

exit 0
