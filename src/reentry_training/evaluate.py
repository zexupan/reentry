import argparse
import torch
from utils import *
import os
from reentry_net import reentry_net
import csv

MAX_INT16 = np.iinfo(np.int16).max

def write_wav(fname, samps, sampling_rate=16000, normalize=True):
    """
    Write wav files in int16, support single/multi-channel
    """
    # for multi-channel, accept ndarray [Nsamples, Nchannels]
    samps = np.divide(samps, np.max(np.abs(samps)))

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



class dataset(data.Dataset):
    def __init__(self,
                mix_lst_path,
                audio_direc,
                visual_direc,
                mixture_direc,
                batch_size=1,
                partition='test',
                sampling_rate=16000,
                mix_no=2,
                pretrained_v=0):

        self.minibatch =[]
        self.audio_direc = audio_direc
        self.visual_direc = visual_direc
        self.mixture_direc = mixture_direc
        self.sampling_rate = sampling_rate
        self.partition = partition
        self.C=mix_no

        mix_csv=open(mix_lst_path).read().splitlines()
        self.mix_lst=list(filter(lambda x: x.split(',')[0]==partition, mix_csv))

    def __getitem__(self, index):
        line = self.mix_lst[index]
        line_cache = line

        mixture_path=self.mixture_direc+self.partition+'/'+ line.replace(',','_').replace('/', '_') +'.wav'
        _, mixture = wavfile.read(mixture_path)
        mixture = self._audio_norm(mixture)
        
        min_length = mixture.shape[0]
        c=0

        visual_path=self.visual_direc+line.split(',')[c*4+1]+'/'+line.split(',')[c*4+2]+'/'+line.split(',')[c*4+3]+'.mp4'
        length = math.floor(min_length/self.sampling_rate*25)

        captureObj = cv.VideoCapture(visual_path)
        roiSequence = []
        roiSize = 112
        mean = 0.4161
        std = 0.1688
        while (captureObj.isOpened()):
            ret, frame = captureObj.read()
            if ret == True:
                grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                grayed = cv.resize(grayed, (roiSize*2,roiSize*2))
                roi = grayed[int(roiSize-(roiSize/2)):int(roiSize+(roiSize/2)), int(roiSize-(roiSize/2)):int(roiSize+(roiSize/2))]
                roiSequence.append(roi)
            else:
                break
        captureObj.release()
        visual = np.asarray(roiSequence)/255.0
        
        visual = (visual[:length] - mean)/std
        if visual.shape[0] < length:
            visual = np.pad(visual, ((int(length - visual.shape[0]),0), (0,0), (0,0) ), mode = 'edge')

        line=line.split(',')
        audio_path=self.audio_direc+line[c*4+1]+'/'+line[c*4+2]+'/'+line[c*4+3]+'.wav'
        _, audio = wavfile.read(audio_path)
        audio = self._audio_norm(audio[:min_length])

        return mixture, audio, visual

    def __len__(self):
        return len(self.mix_lst)

    def _audio_norm(self,audio):
        return np.divide(audio, np.max(np.abs(audio)))

def main(args):
    # Model
    model = reentry_net(args.N, args.L, args.B, args.H, args.P, args.X, args.R,
                        args.C, 800, 256, 0)

    model = model.cuda()
    pretrained_model = torch.load('reentry_model_dict.pt', map_location='cpu')['model']

    state = model.state_dict()
    for key in state.keys():
        pretrain_key = 'module.' + key
        # if pretrain_key in pretrained_model.keys():
        state[key] = pretrained_model[pretrain_key]
    model.load_state_dict(state)


    datasets = dataset(
                mix_lst_path=args.mix_lst_path,
                audio_direc=args.audio_direc,
                visual_direc=args.visual_direc,
                mixture_direc=args.mixture_direc,
                mix_no=args.C)

    test_generator = data.DataLoader(datasets,
            batch_size = 1,
            shuffle = False,
            num_workers = args.num_workers)


    model.eval()
    with torch.no_grad():
        avg_sisnri = 0
        avg_sdri = 0
        avg_pesqi = 0
        avg_stoii = 0
        for i, (a_mix, a_tgt, v_tgt) in enumerate(tqdm.tqdm(test_generator)):
            a_mix = a_mix.cuda().squeeze().float().unsqueeze(0)
            a_tgt = a_tgt.cuda().squeeze().float().unsqueeze(0)
            v_tgt = v_tgt.cuda().squeeze().float().unsqueeze(0)

            est_speaker, estimate_source = model(a_mix, v_tgt)

            sisnr_mix = cal_SISNR(a_tgt, a_mix)
            sisnr_est = cal_SISNR(a_tgt, estimate_source)
            sisnri = sisnr_est - sisnr_mix
            avg_sisnri += sisnri
            print(sisnri)

        
        avg_sisnri = avg_sisnri / (i+1)
        print(avg_sisnri)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("avConv-tasnet")
    
    # Dataloader
    parser.add_argument('--mix_lst_path', type=str, default='/home/panzexu/datasets/voxceleb2/audio_mixture/2_mix_min_800/mixture_data_list_2mix.csv',
                        help='directory including train data')
    parser.add_argument('--audio_direc', type=str, default='/home/panzexu/datasets/voxceleb2/audio_clean/',
                        help='directory including validation data')
    parser.add_argument('--visual_direc', type=str, default='/home/panzexu/datasets/voxceleb2/orig/',
                        help='directory including test data')
    parser.add_argument('--mixture_direc', type=str, default='/home/panzexu/datasets/voxceleb2/audio_mixture/2_mix_min_800/',
                        help='directory of audio')

    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to generate minibatch')

    # Model hyperparameters
    parser.add_argument('--L', default=40, type=int,
                        help='Length of the filters in samples (80=5ms at 16kHZ)')
    parser.add_argument('--N', default=256, type=int,
                        help='Number of filters in autoencoder')
    parser.add_argument('--B', default=256, type=int,
                        help='Number of channels in bottleneck 1 × 1-conv block')
    parser.add_argument('--C', type=int, default=2,
                        help='number of speakers to mix')
    parser.add_argument('--H', default=512, type=int,
                        help='Number of channels in convolutional blocks')
    parser.add_argument('--P', default=3, type=int,
                        help='Kernel size in convolutional blocks')
    parser.add_argument('--X', default=8, type=int,
                        help='Number of convolutional blocks in each repeat')
    parser.add_argument('--R', default=4, type=int,
                        help='Number of repeats')

    args = parser.parse_args()

    main(args)