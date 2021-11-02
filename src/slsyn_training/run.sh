#!/bin/sh

gpu_id=6

continue_from=

if [ -z ${continue_from} ]; then
	log_name='avConv_'$(date '+%Y-%m-%d(%H:%M:%S)')
	mkdir logs
	mkdir logs/$log_name
else
	log_name=${continue_from}
fi

CUDA_VISIBLE_DEVICES="$gpu_id" \
python -W ignore \
-m torch.distributed.launch \
--nproc_per_node=1 \
--master_port=3396 \
main.py \
\
--log_name $log_name \
\
--audio_direc '/home/panzexu/datasets/voxceleb2/audio_sync/' \
--visual_direc '/home/panzexu/datasets/voxceleb2/orig/' \
--mix_lst_path '/home/panzexu/datasets/voxceleb2/audio_sync/sync_list.csv' \
--epochs 200 \
--num_workers 8 \
\
--batch_size 16 \
\
--use_tensorboard 1 \
>logs/$log_name/console2.txt 2>&1

# --continue_from ${continue_from} \
