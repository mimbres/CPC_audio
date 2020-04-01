# Copyright (c) Cochlear.ai, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Speaker recognition experiment with libris-selection-dataset from SincNet.

Dependency:
    - cpc.eval.sungkyun_libri_sel_dataloader.py
    - cpc.eval.sungkyun_classifier.py
    - cpc.eval.utils.file_utils.py
    - cpc.eval.utils.sampler.py
    - cpc.utils.misc.py
    - cpc.feature_loader.py
    
Created on Wed Apr  1 14:57:05 2020
@author: skchang@cochlear.ai
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cpc.utils.misc as utils
import cpc.feature_loader as fl
from cpc.eval.sungkyun_classifier import MLP, MobileNetV2 
from cpc.eval.sungkyun_libri_sel_dataloader import LibriSelectionDataset
torch.multiprocessing.set_sharing_strategy('file_system') 
# This was required for preventing multiprocessing errors.


"""Config."""
SEL_FEAT = 'cz' # or 'c' or 'z'
CPC_CHECKPOINT_PATH = '../exp_100_lstm_transformer_unsup/'
MAX_EPOCHS = 200

DB_WAV_ROOT = './Speaker_ID_DB/Libri-from-sincnet/Librispeech_spkid_sel/'
TR_LIST_PATH = './Speaker_ID_DB/Libri-from-sincnet/libri_tr.scp'
TS_LIST_PATH = './Speaker_ID_DB/Libri-from-sincnet/libri_te.scp'
LABEL_PATH = './Speaker_ID_DB/Libri-from-sincnet/libri_dict.npy'
nGPU = torch.cuda.device_count()


"""Data loading..."""
db_train = LibriSelectionDataset(sizeWindow=20480, db_wav_root=DB_WAV_ROOT, 
                                 fps_list=TR_LIST_PATH, label_path=LABEL_PATH,
                                 n_process_loader=4, MAX_SIZE_LOADED=400000000)
train_loader = db_train.getDataLoader(batchSize=256, type='sequqential',
                                     randomOffset=False, numWorkers=0)


"""Load model: c, z, label = feat_gen(raw_audio, label)"""
feat_gen, d_c, d_z = fl.loadModel([CPC_CHECKPOINT_PATH], loadStateDict=False)

if nGPU > 0:
    feat_gen = feat_gen.cuda()
    feat_gen = torch.nn.DataParallel(feat_gen, device_ids=range(nGPU))
feat_gen = feat_gen.eval()


"""Create classifier: clf()"""
d_feat = int('c' in SEL_FEAT) * d_c + int('z' in SEL_FEAT) * d_z
n_classes = len(db_train.speakers)
clf = MLP(d_feat, 2048, n_classes)
if nGPU > 0:
    clf = clf.cuda()
    clf = torch.nn.DataParallel(clf, device_ids=range(nGPU))

"""Training methods."""
def train_step(feat_gen, clf, train_loader, optimizer, ep='unknown'):
    feat_gen = feat_gen.eval()
    clf = clf.train()
    log_smax = nn.LogSoftmax(dim=2)
    loss_func = nn.NLLLoss()
    for i, (batch_data, labels) in enumerate(train_loader):
        #if i>3: break; # batch_data:(B,1,20480), labels:(B)
        c, z, _ = feat_gen(batch_data, None) # c:(B,128,256), z:(B:128,256) (B,T,F)
        c, z = c.detach(), z.detach()
        
        c = c[:,-28:, :]
        z = z[:,-28:, :]
        
        # Concat feat
        feat = [] 
        if ('c' in SEL_FEAT): feat.append(c) 
        if ('z' in SEL_FEAT): feat.append(z) 
        feat = torch.cat(feat, dim=2)
        feat.requires_grad=False
        
        # Predict --> loss
        optimizer.zero_grad()
        
        logit = clf(feat) # logit: (B,T,Class)
        logit = log_smax(logit)
        logit = torch.mean(logit, dim=1) # avg.logit: (B, Class)
        loss = loss_func(logit, labels.cuda())
        loss.backward()
        
        optimizer.step()
        print(f'Ep={ep}, Iter={i}, loss={loss.item()}')
    return loss.item()




"""Train."""
optimizer = torch.optim.Adam(clf.parameters(), lr=0.001)
for ep in range(MAX_EPOCHS):
    logs_train = train_step(feat_gen, clf, train_loader, optimizer, ep)
    #logs_val = val_step(feature_maker, criterion, val_loader)
    #logs = {"epoch": [], "iter": [], "saveStep": -1} # saveStep=-1, save only best checkpoint!
    
