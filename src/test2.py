#!/usr/bin/python3
#coding=utf-8

import os
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dataset
from z230321_B_ALC_s import PGNet

class Test(object):
    def __init__(self, Dataset, Network, path, model):
        ## dataset
        self.model  = model
        self.cfg    = Dataset.Config(datapath=path, snapshot=model, mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net    = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()

    def save(self):
        with torch.no_grad():
            for image, mask, shape, name in self.loader:
                image = image.cuda().float()
                mask  = mask.cuda().float()
                p = self.net(image, shape=shape)
                out_resize   = F.interpolate(p[0],size=shape, mode='bilinear')
                #out_resize   = F.interpolate(p,size=shape, mode='bilinear')
                pred   = torch.sigmoid(out_resize[0,0])
                pred  = (pred*255).cpu().numpy()
                head  = '../result/'+self.model+'/'+ self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))

if __name__=='__main__':
    #for path in ['../data/DIS-TE1','../data/DIS-TE2','../data/DIS-TE3','../data/DIS-TE14','../data/DIS-VD']:
    for path in ['../data/DIS-TE1','../data/DIS-TE2','../data/DIS-TE3','../data/DIS-TE4','../data/DIS-VD']:
	    #for model in ['model-28','model-29','model-30','model-31','model-32']:
	    for model in ['model-32']:
                t = Test(dataset,PGNet, path,'/home/living/TYY/DIS/MyPGNet/model/z230321_B_ALC_s/'+model)
                t.save()
