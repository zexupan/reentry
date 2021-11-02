#!/bin/bash 

direc=/home/panzexu/datasets/voxceleb2/

data_direc=${direc}orig/

train_samples=4000000 # no. of train mixture samples simulated
test_samples=40000 # no. of test mixture samples simulated
mix_db=10 # random db ratio from -10 to 10db
sync_list=sync_list_2mix.csv #mixture datalist
sampling_rate=16000 # audio sampling rate
min_length=1 # minimum length of audio
max_length=4 # minimum length of audio

audio_data_direc=${direc}audio_clean/ # Target audio saved directory
sync_audio_direc=${direc}audio_sync/ # Audio mixture saved directory

stage 1: Remove repeated datas in pretrain and train set, extract audio from mp4, create mixture list
echo 'stage 1: create mixture list'
python 1_create_list.py \
--data_direc $data_direc \
--mix_db $mix_db \
--train_samples $train_samples \
--test_samples $test_samples \
--audio_data_direc $audio_data_direc \
--min_length $min_length \
--max_length $max_length \
--sampling_rate $sampling_rate \
--sync_list $sync_list \
--sync_audio_direc $sync_audio_direc \

