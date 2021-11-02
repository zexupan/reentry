import numpy as np
import math
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.utils.data as data
import scipy.io.wavfile as wavfile
from itertools import permutations
from apex import amp
import tqdm
import os
import cv2 as cv
import random

random.seed(0)

EPS = 1e-6

class dataset(data.Dataset):
    def __init__(self,
                mix_lst_path,
                audio_direc,
                visual_direc,
                batch_size,
                partition='test',
                sampling_rate=16000,
                max_length=4):

        self.minibatch =[]
        self.audio_direc = audio_direc
        self.visual_direc = visual_direc
        self.sampling_rate = sampling_rate
        self.partition = partition
        self.max_length = max_length
        self.batch_size = batch_size 
        self.mean = 0.4161
        self.std = 0.1688

        mix_lst=open(mix_lst_path).read().splitlines()
        mix_lst=list(filter(lambda x: x.split(',')[0]==partition, mix_lst))
        mix_lst = [ x for x in mix_lst if "id08137,Pvrmbe76RkU/00196" not in x ]
        mix_lst = [ x for x in mix_lst if "train,sync_m,train,id08497,FMxZ6xo_fAA/00085,32640,50560,0,train,id06017,CNVpuyN_e94/00061,4774,22694" not in x ]

        random.shuffle(mix_lst)

        # mix_lst = mix_lst[:400000]
        # if partition=='test':
        #     mix_lst = mix_lst[:4000]

        sorted_mix_lst = sorted(mix_lst, key=lambda data: (int(data.split(',')[6]) - int(data.split(',')[5]) ) , reverse=True)

        start = 0
        while True:
            end = min(len(sorted_mix_lst), start + self.batch_size)
            self.minibatch.append(sorted_mix_lst[start:end])
            if end == len(sorted_mix_lst):
                break
            start = end

    def __getitem__(self, index):
        batch_lst = self.minibatch[index]
        min_length = int(batch_lst[-1].split(',')[6]) - int(batch_lst[-1].split(',')[5])

        audios=[]
        visuals=[]
        labels=[]
        roiSize = 112
        for line in batch_lst:
            # read audio
            audio_path=self.audio_direc+line.split(',')[0]+'/'+line.split(',')[1]+'/'+ line.replace(',','_').replace('/','_')+'.wav'
            line=line.split(',')
            _, audio = wavfile.read(audio_path)
            audio = self._audio_norm(audio[:min_length])
            audios.append(audio)
            
            # read visual
            visual_offset = int(line[7])
            if visual_offset == 0:
                labels.append(0)
            else: labels.append(1)

            start = int(np.round(int(line[5]) / 640.0 + visual_offset))
            end = int(np.round(start + min_length/640.0))

            visual_path=self.visual_direc+line[2]+'/'+line[3]+'/'+line[4]+'.mp4'
            captureObj = cv.VideoCapture(visual_path)
            roiSequence = []
            while (captureObj.isOpened()):
                ret, frame = captureObj.read()
                if ret == True or len(roiSequence) < end:
                    grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    grayed = cv.resize(grayed, (roiSize*2,roiSize*2))
                    roi = grayed[int(roiSize-(roiSize/2)):int(roiSize+(roiSize/2)), int(roiSize-(roiSize/2)):int(roiSize+(roiSize/2))]
                    roiSequence.append(roi)
                else:
                    break
            captureObj.release()
            images = np.asarray(roiSequence)/255.0
            
            images = (images[start:end] - self.mean)/self.std
            visuals.append(images)
    
        return np.asarray(audios)[...,:self.max_length*self.sampling_rate], \
                np.asarray(visuals)[...,:self.max_length*25,:,:], \
                np.asarray(labels)

    def __len__(self):
        return len(self.minibatch)

    def _audio_norm(self,audio):
        return np.divide(audio, np.max(np.abs(audio)))


class DistributedSampler(data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # indices = torch.randperm(len(self.dataset), generator=g).tolist()
            ind = torch.randperm(int(len(self.dataset)/self.num_replicas), generator=g)*self.num_replicas
            indices = []
            for i in range(self.num_replicas):
                indices = indices + (ind+i).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        # indices = indices[self.rank:self.total_size:self.num_replicas]
        indices = indices[self.rank*self.num_samples:(self.rank+1)*self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def get_dataloader(args, partition):
    datasets = dataset(
                mix_lst_path=args.mix_lst_path,
                audio_direc=args.audio_direc,
                visual_direc=args.visual_direc,
                batch_size=args.batch_size,
                max_length=args.max_length,
                partition=partition)

    sampler = DistributedSampler(
        datasets,
        num_replicas=args.world_size,
        rank=args.local_rank) if args.distributed else None

    generator = data.DataLoader(datasets,
            batch_size = 1,
            shuffle = (sampler is None),
            num_workers = args.num_workers,
            sampler=sampler)

    return sampler, generator

@amp.float_function
def cal_SISNR(source, estimate_source):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        source: torch tensor, [batch size, sequence length]
        estimate_source: torch tensor, [batch size, sequence length]
    Returns:
        SISNR, [batch size]
    """
    assert source.size() == estimate_source.size()

    # Step 1. Zero-mean norm
    source = source - torch.mean(source, axis = -1, keepdim=True)
    estimate_source = estimate_source - torch.mean(estimate_source, axis = -1, keepdim=True)

    # Step 2. SI-SNR
    # s_target = <s', s>s / ||s||^2
    ref_energy = torch.sum(source ** 2, axis = -1, keepdim=True) + EPS
    proj = torch.sum(source * estimate_source, axis = -1, keepdim=True) * source / ref_energy
    # e_noise = s' - s_target
    noise = estimate_source - proj
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    ratio = torch.sum(proj ** 2, axis = -1) / (torch.sum(noise ** 2, axis = -1) + EPS)
    sisnr = 10 * torch.log10(ratio + EPS)

    return sisnr


if __name__ == '__main__':
    datasets = dataset(
                mix_lst_path='/home/panzexu/datasets/voxceleb2/audio_sync/huge/sync_list.csv',
                audio_direc='/home/panzexu/datasets/voxceleb2/audio_sync/huge/',
                visual_direc='/home/panzexu/datasets/voxceleb2/orig/',
                partition='train',
                batch_size=1)
    data_loader = data.DataLoader(datasets,
                batch_size = 1,
                shuffle= True,
                num_workers = 24)

    for a_mix , visual, label in tqdm.tqdm(data_loader):
        # print(a_mix.size())
        # print(visual.size())
        # print(label.size())
        pass

    # a = np.ones((24,512))
    # print(a.shape)
    # a = np.pad(a, ((0,-1), (0,0)), 'edge')
    # print(a.shape)

    # a = np.random.rand(2,2,3)
    # print(a)
    # a = a.reshape(4,3)
    # print(a)
