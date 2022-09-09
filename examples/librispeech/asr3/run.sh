#!/bin/bash
set -e

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

gpus=1
stage=2
stop_stage=3
conf_path=conf/wav2vec2ASR.yaml
ips=            #xx.xx.xx.xx,xx.xx.xx.xx
decode_conf_path=conf/tuning/decode.yaml
avg_num=1
audio_file=data/demo_002_en.wav
dict_path=data/lang_char/vocab.txt
dp_log='12batch'
. ${MAIN_ROOT}/utils/parse_options.sh || exit 1;

avg_ckpt=avg_${avg_num}
ckpt=12batch
echo "checkpoint name ${ckpt}"
ckpt_prefix=exp/${ckpt}/checkpoints/${avg_ckpt}

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    bash ./local/data.sh || exit -1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # train model, all `ckpt` under `exp` dir
    CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${ckpt} ${dp_log} ${ips} 
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # avg n best model
    avg.sh best exp/${ckpt}/checkpoints ${avg_num}
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # attetion resocre decoder
    echo ${ckpt_prefix}
    ./local/test.sh ${conf_path} ${decode_conf_path} ${dict_path} ${ckpt_prefix} || exit -1
fi
