#!/bin/sh

gpu_id=0
continue_from=
if [ -z ${continue_from} ]; then
	log_name='avaNet_'$(date '+%Y-%m-%d(%H:%M:%S)')
	mkdir logs
	mkdir logs/$log_name
else
	log_name=${continue_from}
fi

CUDA_VISIBLE_DEVICES="$gpu_id" \
python -W ignore \
-m torch.distributed.launch \
--nproc_per_node=1 \
--master_port=3197 \
main.py \
\
--log_name $log_name \
\
--audio_direc '/home/panzexu/datasets/voxceleb2/audio_clean/' \
--visual_direc '/home/panzexu/datasets/voxceleb2/visual_embedding/sync/sync_av/' \
--pretrained_v 0 \
--mix_lst_path '/home/panzexu/datasets/voxceleb2/audio_mixture/2_mix_min_800/mixture_data_list_2mix.csv' \
--mixture_direc '/home/panzexu/datasets/voxceleb2/audio_mixture/2_mix_min_800/' \
--C 2 \
--epochs 300 \
\
--effec_batch_size 8 \
--accu_grad 1 \
--batch_size 2 \
--num_workers 4 \
--use_tensorboard 1 \
>logs/$log_name/console.txt 2>&1


# --continue_from ${continue_from} \

# --visual_direc '/home/panzexu/datasets/voxceleb2/visual_embedding/sync/sync_av/' \
