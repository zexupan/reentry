import argparse
import torch
import torch.utils.data as data
import os
import numpy as np
import tqdm
import math
import scipy.io.wavfile as wavfile
from slsyn_net import slsyn_net
import cv2 as cv

class dataset(data.Dataset):
    def __init__(self,
                mix_lst_path,
                audio_direc,
                visual_direc,
                batch_size=1,
                partition='test',
                sampling_rate=16000):

        self.mixture_direc = audio_direc
        self.visual_direc = visual_direc
        self.sampling_rate = sampling_rate

        self.mix_lst=open(mix_lst_path).read().splitlines()
        self.mix_lst=list(filter(lambda x: x.split(',')[0]==partition, self.mix_lst))

    def __getitem__(self, index):
        line = self.mix_lst[index]
        embedding_save_path=line.split(',')[0]+'/'+ line.replace(',','_').replace('/', '_')

        mixture_path=self.mixture_direc+line.split(',')[0]+'/'+ line.replace(',','_').replace('/', '_') +'.wav'
        _, mixture = wavfile.read(mixture_path)
        mixture = self._audio_norm(mixture)
        
        min_length = mixture.shape[0]

        line=line.split(',')

        c = 0 # The first speaker in the mixture list is the target speaker
        # read video
        length = math.floor(min_length/self.sampling_rate*25)

        roiSize = 112
        visual_path=self.visual_direc+line[c*4+1]+'/'+line[c*4+2]+'/'+line[c*4+3]+'.mp4'
        captureObj = cv.VideoCapture(visual_path)
        roiSequence = []
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
        visual = (visual[:length] - 0.4161)/0.1688

        if visual.shape[0] < length:
            visual = np.pad(visual, ((0,int(length - visual.shape[0])),(0,0),(0,0)), mode = 'edge')

        return mixture, visual, (embedding_save_path)

    def __len__(self):
        return len(self.mix_lst)

    def _audio_norm(self,audio):
        return np.divide(audio, np.max(np.abs(audio)))

def main(args):
    # Model
    model = slsyn_net()
    model = model.cuda()
    pretrained_model = torch.load('slsyn_model_dict.pt', map_location='cpu')['model']

    state = model.state_dict()
    for key in state.keys():
        pretrain_key = 'module.' + key
        if pretrain_key in pretrained_model.keys():
            state[key] = pretrained_model[pretrain_key]
        else:
            print("not %s loaded" % pretrain_key)
    model.load_state_dict(state)



    datasets = dataset(
                mix_lst_path=args.mix_lst_path,
                audio_direc=args.audio_direc,
                visual_direc=args.visual_direc)

    test_generator = data.DataLoader(datasets,
            batch_size = 1,
            shuffle = False,
            num_workers = 1)

    
    model.eval()
    with torch.no_grad():
        for i, (a_mix, v_tgt, fname) in enumerate(tqdm.tqdm(test_generator)):
            a_mix = a_mix.cuda().float()
            v_tgt = v_tgt.cuda().float()

            out = model(a_mix, v_tgt)

            out = out.squeeze(0).cpu().numpy().T
            # print(out.shape)

            save_path = "/home/panzexu/datasets/voxceleb2/visual_embedding/sync/eval/sync_av/" + fname[0] +'.npy'

            if not os.path.exists(save_path.rsplit('/', 1)[0]):
                os.makedirs(save_path.rsplit('/', 1)[0])
            np.save(save_path, out)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser("SLSyn network extract embedding")
    
    # Dataloader
    parser.add_argument('--mix_lst_path', type=str, default='/home/panzexu/datasets/voxceleb2/audio_mixture/2_mix_min_800/mixture_data_list_2mix.csv',
                        help='directory including train data')
    parser.add_argument('--visual_direc', type=str, default='/home/panzexu/datasets/voxceleb2/orig/',
                        help='directory including test data')
    parser.add_argument('--audio_direc', type=str, default='/home/panzexu/datasets/voxceleb2/audio_mixture/2_mix_min_800/',
                        help='directory of audio')
    args = parser.parse_args()

    main(args)