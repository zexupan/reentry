import os
import numpy as np 
import argparse
import csv
import tqdm
import librosa
import scipy.io.wavfile as wavfile
from multiprocessing import Pool

np.random.seed(0)

MAX_INT16 = np.iinfo(np.int16).max

def read_wav(fname, normalize=True):
    """
    Read wave files using scipy.io.wavfile(support multi-channel)
    """
    # samps_int16: N x C or N
    #   N: number of samples
    #   C: number of channels
    sampling_rate, samps_int16 = wavfile.read(fname)
    # N x C => C x N
    samps = samps_int16.astype(np.float)
    # tranpose because I used to put channel axis first
    if samps.ndim != 1:
        samps = np.transpose(samps)
    # normalize like MATLAB and librosa
    if normalize:
        samps = samps / MAX_INT16
    return sampling_rate, samps

def write_wav(fname, samps, sampling_rate=16000, normalize=True):
	"""
	Write wav files in int16, support single/multi-channel
	"""
	# for multi-channel, accept ndarray [Nsamples, Nchannels]
	if samps.ndim != 1 and samps.shape[0] < samps.shape[1]:
		samps = np.transpose(samps)
		samps = np.squeeze(samps)
	# same as MATLAB and kaldi
	if normalize:
		samps = samps * MAX_INT16
		samps = samps.astype(np.int16)
	fdir = os.path.dirname(fname)
	if fdir and not os.path.exists(fdir):
		os.makedirs(fdir)
	# NOTE: librosa 0.6.0 seems could not write non-float narray
	#       so use scipy.io.wavfile instead
	wavfile.write(fname, sampling_rate, samps)

def main(args):
	# read the datalist and separate into train, val and test set
	train_list=[]
	test_list=[]

	print("Gathering file names")

	# Get test set list of audios
	for path, dirs ,files in os.walk(args.audio_data_direc + 'test/'):
		for filename in files:
			if filename[-4:] =='.wav':
				ln = [path.split('/')[-3], path.split('/')[-2], path.split('/')[-1] +'/'+ filename.split('.')[0]]
				test_list.append(ln)

	# Get train set list of audios
	for path, dirs ,files in os.walk(args.audio_data_direc + 'train/'):
		for filename in files:
			if filename[-4:] =='.wav':
				ln = [path.split('/')[-3], path.split('/')[-2], path.split('/')[-1] +'/'+ filename.split('.')[0]]
				train_list.append(ln)


	f=open(args.sync_list,'w')
	w=csv.writer(f)

	sampling_ratio = 640.0 #16000/25

	for data_list in [test_list, train_list]:
		if len(data_list) < 40000:
			partition = 'test'
			no_po_clean = 5000
			no_po_mix = 15000
			no_ng_clean = 5000
			no_ng_mix = 15000
		else:
			partition = 'train'
			no_po_clean = 500000
			no_po_mix = 1500000
			no_ng_clean = 500000
			no_ng_mix = 1500000

		# create positive clean data
		counter = 0
		cat = 'sync_s'
		while counter < no_po_clean:
			idx = np.random.randint(0, len(data_list))
			ln = data_list[idx]
			_, audio_sample=read_wav(args.audio_data_direc+ln[0]+'/'+ln[1]+'/'+ln[2]+'.wav')
			duration = int(np.round(np.random.randint(args.min_length * 25, args.max_length*25)*sampling_ratio))
			excess = int(np.floor((audio_sample.shape[0]-duration) /sampling_ratio))
			if excess < 2:
				continue
			start = int(np.round(np.random.randint(0, excess) * sampling_ratio))
			end = start + duration
			audio = audio_sample[start:end]
			save_list = [partition,cat] + ln +[str(start), str(end), str(0)]
			mixture_save_path = '_'.join(save_list).replace('/','_')
			w.writerow(save_list)
			audio = np.divide(audio, np.max(np.abs(audio)))
			assert audio.shape[0] == duration, print(audio.shape)
			full_mixture_save_path = args.sync_audio_direc+ partition +'/' + cat +'/' + mixture_save_path +'.wav'
			write_wav(full_mixture_save_path, audio)
			counter +=1

		# create positive mix data
		counter = 0
		cat = 'sync_m'
		while counter < no_po_mix:
			idx = np.random.randint(0, len(data_list))
			ln = data_list[idx]
			_, audio_sample=read_wav(args.audio_data_direc+ln[0]+'/'+ln[1]+'/'+ln[2]+'.wav')
			duration = int(np.round(np.random.randint(args.min_length * 25, args.max_length*25)*sampling_ratio))
			excess = int(np.floor((audio_sample.shape[0]-duration) /sampling_ratio))
			if excess < 2:
				continue
			start = int(np.round(np.random.randint(0, excess) * sampling_ratio))
			end = start + duration
			audio = audio_sample[start:end]
			target_power = np.linalg.norm(audio, 2)**2 / audio.size
			save_list = [partition,cat] + ln +[str(start), str(end), str(0)]
			mix_add = 1

			m = np.random.randint(2,3)
			while mix_add < m:
				idx = np.random.randint(0, len(data_list))
				ln = data_list[idx]
				_, audio_sample=read_wav(args.audio_data_direc+ln[0]+'/'+ln[1]+'/'+ln[2]+'.wav')
				if duration > (audio_sample.shape[0]-2):
					continue
				start = np.random.randint(0, audio_sample.shape[0] - duration)
				db_ratio = np.random.uniform(-args.mix_db,args.mix_db)

				infef_audio = audio_sample[start: start + duration]
				intef_power = np.linalg.norm(infef_audio, 2)**2 / infef_audio.size


				scalar = (10**(db_ratio/20)) * np.sqrt(target_power/intef_power)
				audio += infef_audio * scalar

				save_list = save_list+ ln +[str(start), str(start + duration), str(db_ratio)]
				mix_add +=1

			mixture_save_path = '_'.join(save_list).replace('/','_')
			w.writerow(save_list)
			audio = np.divide(audio, np.max(np.abs(audio)))
			assert audio.shape[0] == duration, print(audio.shape)
			full_mixture_save_path = args.sync_audio_direc+ partition +'/' + cat +'/' + mixture_save_path +'.wav'
			write_wav(full_mixture_save_path, audio)
			counter +=1

		# create negative clean data
		counter = 0
		cat = 'unsync_s'
		while counter < no_ng_clean:
			while True:
				video_shift = np.random.randint(-25, 26)
				if video_shift > 5 or video_shift <-5:
					break
			idx = np.random.randint(0, len(data_list))
			ln = data_list[idx]
			_, audio_sample=read_wav(args.audio_data_direc+ln[0]+'/'+ln[1]+'/'+ln[2]+'.wav')
			duration = int(np.round(np.random.randint(args.min_length * 25, args.max_length*25)*sampling_ratio))
			
			excess = int(np.floor((audio_sample.shape[0]-duration) /sampling_ratio))
			if excess < 60:
				continue
			start = int(np.round(np.random.randint(28, excess-30) * sampling_ratio))
			end = start + duration
			audio = audio_sample[start:end]
			save_list = [partition,cat] + ln +[str(start), str(end), str(video_shift)] 
			mixture_save_path = '_'.join(save_list).replace('/','_')
			w.writerow(save_list)
			audio = np.divide(audio, np.max(np.abs(audio)))
			assert audio.shape[0] == duration, print(audio.shape)
			full_mixture_save_path = args.sync_audio_direc+ partition +'/' + cat +'/' + mixture_save_path +'.wav'
			write_wav(full_mixture_save_path, audio)
			counter +=1



		# create negative mix data
		counter = 0
		cat = 'unsync_m'
		while counter < no_ng_mix:
			while True:
				video_shift = np.random.randint(-25, 26)
				if video_shift >5 or video_shift <-5:
					break
			idx = np.random.randint(0, len(data_list))
			ln = data_list[idx]
			_, audio_sample=read_wav(args.audio_data_direc+ln[0]+'/'+ln[1]+'/'+ln[2]+'.wav')
			duration = int(np.round(np.random.randint(args.min_length * 25, args.max_length*25)*sampling_ratio))

			excess = int(np.floor((audio_sample.shape[0]-duration) /sampling_ratio))
			if excess < 60:
				continue
			start = int(np.round(np.random.randint(28, excess-30) * sampling_ratio))
			end = start + duration
			audio = audio_sample[start:end]
			target_power = np.linalg.norm(audio, 2)**2 / audio.size
			save_list = [partition,cat] + ln +[str(start), str(end), str(video_shift)]
			mix_add = 1

			m = np.random.randint(2,3)

			while mix_add < m:
				idx = np.random.randint(0, len(data_list))
				ln = data_list[idx]
				_, audio_sample=read_wav(args.audio_data_direc+ln[0]+'/'+ln[1]+'/'+ln[2]+'.wav')
				if duration > (audio_sample.shape[0]-2):
					continue
				start = np.random.randint(0, audio_sample.shape[0] - duration)
				db_ratio = np.random.uniform(-args.mix_db,args.mix_db)

				infef_audio = audio_sample[start: start + duration]
				intef_power = np.linalg.norm(infef_audio, 2)**2 / infef_audio.size

				scalar = (10**(db_ratio/20)) * np.sqrt(target_power/intef_power)
				audio += infef_audio * scalar

				save_list = save_list+ ln +[str(start), str(start + duration), str(db_ratio)]
				mix_add +=1

			mixture_save_path = '_'.join(save_list).replace('/','_')
			w.writerow(save_list)
			audio = np.divide(audio, np.max(np.abs(audio)))
			assert audio.shape[0] == duration, print(audio.shape)
			full_mixture_save_path = args.sync_audio_direc+ partition +'/' + cat +'/' + mixture_save_path +'.wav'
			write_wav(full_mixture_save_path, audio)
			counter +=1


	f.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='LRS2 dataset')
	parser.add_argument('--data_direc', type=str)
	parser.add_argument('--mix_db', type=float)
	parser.add_argument('--train_samples', type=int)
	parser.add_argument('--test_samples', type=int)
	parser.add_argument('--audio_data_direc', type=str)
	parser.add_argument('--min_length', type=int)
	parser.add_argument('--max_length', type=int)
	parser.add_argument('--sampling_rate', type=int)
	parser.add_argument('--sync_list', type=str)
	parser.add_argument('--sync_audio_direc', type=str)
	args = parser.parse_args()
	
	main(args)