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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cpc.utils.misc as utils
import cpc.feature_loader as fl
from cpc.eval.sungkyun_classifier import MLP, MobileNetV2, SpeakerClf 
#from cpc.eval.sungkyun_libri_sel_dataloader import LibriSelectionDataset
from cpc.dataset import AudioBatchData, findAllSeqs, filterSeqs, parseSeqLabels
torch.multiprocessing.set_sharing_strategy('file_system') 
# This was required for preventing multiprocessing errors.


"""Config."""
SEL_FEAT = 'c' # or 'c' or 'z'
CPC_CHECKPOINT_PATH = '../exp_100_gru_linear/checkpoint_95.pt' #../exp_100_lstm_transformer_unsup/'
MAX_EPOCHS = 2000


nGPU = torch.cuda.device_count()


"""Data loading..."""
seqNames, speakers = findAllSeqs('../../../CPC_librispeech/dataset/LibriSpeech/train-clean-100/',
                                 extension=".flac",
                                 loadCache=False)
seqTrain=[]; seqTest=[];
cnt_spk = np.zeros(len(speakers))
for seqn in seqNames:
    spk_id, fname = seqn
    cnt_spk[spk_id] += 1
    if cnt_spk[spk_id] > 60:
        seqTest.append(seqn)
    else:
        seqTrain.append(seqn)
    
    
# seq_train = filterSeqs(args.pathTrain, seqNames)
# seq_val = filterSeqs(args.pathVal, seqNames)

db_train = AudioBatchData(path='../../../CPC_librispeech/dataset/LibriSpeech/train-clean-100/',
                     sizeWindow=20480,
                     seqNames=seqTrain,
                     phoneLabelsDict=None,
                     nSpeakers=len(speakers),
                     nProcessLoader=50,
                     MAX_SIZE_LOADED=400000000)
train_loader = db_train.getDataLoader(batchSize=64, type='uniform', #'sequential',
                                     randomOffset=False, numWorkers=0)

db_test = AudioBatchData(path='../../../CPC_librispeech/dataset/LibriSpeech/train-clean-100/',
                     sizeWindow=20480,
                     seqNames=seqTest,
                     phoneLabelsDict=None,
                     nSpeakers=len(speakers),
                     nProcessLoader=50,
                     MAX_SIZE_LOADED=400000000)
test_loader = db_test.getDataLoader(batchSize=64, type='sequential',
                                     randomOffset=False, numWorkers=0)


"""Load model: c, z, label = feat_gen(raw_audio, label)"""
feat_gen, d_c, d_z = fl.loadModel([CPC_CHECKPOINT_PATH], loadStateDict=True)

if nGPU > 0:
    feat_gen = feat_gen.cuda()
    feat_gen = torch.nn.DataParallel(feat_gen, device_ids=range(nGPU))
feat_gen.optimize = False
feat_gen.eval()
for g in feat_gen.parameters():
    g.requires_grad = False


"""Create classifier: clf()"""
d_feat = int('c' in SEL_FEAT) * d_c + int('z' in SEL_FEAT) * d_z
n_classes = len(db_train.speakers)
#clf = MLP(d_feat, 2048, n_classes)
clf = SpeakerClf(d_feat, n_classes)
if nGPU > 0:
    clf = clf.cuda()
    clf = torch.nn.DataParallel(clf, device_ids=range(nGPU))

"""Training methods."""
# def train_step(feat_gen, clf, train_loader, optimizer, ep='unknown'):
#     feat_gen = feat_gen.eval()
#     clf = clf.train()
#     log_smax = nn.LogSoftmax(dim=2)
#     loss_func = nn.NLLLoss()
#     for i, (batch_data, labels) in enumerate(train_loader):
#         #if i>3: break; # batch_data:(B,1,20480), labels:(B)
#         c, z, _ = feat_gen(batch_data, None) # c:(B,128,256), z:(B:128,256) (B,T,F)
#         c, z = c.detach(), z.detach()
        
#         c = c[:,-1, :].unsqueeze(dim=1)
#         z = z[:,-1:, :].unsqueeze(dim=1)
        
#         # Concat feat
#         feat = [] 
#         if ('c' in SEL_FEAT): feat.append(c) 
#         if ('z' in SEL_FEAT): feat.append(z) 
#         feat = torch.cat(feat, dim=2)
#         feat.requires_grad=False
        
#         # Predict --> loss
#         optimizer.zero_grad()
        
#         logit = clf(feat) # logit: (B,T,Class)
#         logit = log_smax(logit)
#         logit = torch.mean(logit, dim=1) # avg.logit: (B, Class)
#         loss = loss_func(logit, labels.cuda())
#         loss.backward()
        
#         optimizer.step()
#         #print(f'Ep={ep}, Iter={i}, loss={loss.item()}')
#     return loss.item()
def train_step(feat_gen, clf, train_loader, optimizer, ep='unknown'):
    feat_gen.eval()
    clf.train()
    #loss_func = nn.CrossEntropyLoss()
    logs = {"locLoss_train": 0,  "locAcc_train": 0}
    for i, (batch_data, labels) in enumerate(train_loader):
        #if i>3: break; # batch_data:(B,1,20480), labels:(B)
        c, z, _ = feat_gen(batch_data, None) # c:(B,128,256), z:(B:128,256) (B,T,F)
        c = c.detach()
        z = z.detach()
        #c = c[:,-1, :]
        
        # Predict --> loss
        optimizer.zero_grad()
        
        #logit = clf(c) # logit: (B,Class)
        #loss = loss_func(logit, labels.cuda()).view(1, -1)
        all_losses, all_acc = clf(c, None, labels)
        totLoss = all_losses.sum()
        totLoss.backward() 
        optimizer.step()
        #acc = (logit.detach().cpu().max(1)[1] == labels).double().mean().view(1, -1)
        logs["locLoss_train"] += np.asarray([all_losses.mean().item()])
        logs["locAcc_train"] += np.asarray([all_acc.mean().item()])

    logs = utils.update_logs(logs, i)
    logs["iter"] = i

    return logs["locLoss_train"], logs["locAcc_train"]


def test_step(feat_gen, clf, test_loader, optimizer, ep='unknown'):
    feat_gen.eval()
    clf.eval()

    logs = {"locLoss_test": 0,  "locAcc_test": 0}
    for i, (batch_data, labels) in enumerate(train_loader):
        #if i>3: break; # batch_data:(B,1,20480), labels:(B)
        c, z, _ = feat_gen(batch_data, None) # c:(B,128,256), z:(B:128,256) (B,T,F)
        c = c.detach()
        z = z.detach()
        
        # Predict --> loss
        all_losses, all_acc = clf(c, None, labels)
        logs["locLoss_test"] += np.asarray([all_losses.mean().item()])
        logs["locAcc_test"] += np.asarray([all_acc.mean().item()])

    logs = utils.update_logs(logs, i)
    logs["iter"] = i

    return logs["locLoss_test"], logs["locAcc_test"]
        

"""Train."""
optimizer = torch.optim.Adam(list(clf.parameters()), lr=2e-4, betas=(0.9, 0.999), eps=2e-8)
for ep in range(MAX_EPOCHS):
    loss_train, acc_train = train_step(feat_gen, clf, train_loader, optimizer, ep)
    #logs_val = val_step(feature_maker, criterion, val_loader)
    #logs = {"epoch": [], "iter": [], "saveStep": -1} # saveStep=-1, save only best checkpoint!
    print(f'Ep={ep}, loss={loss_train}, acc={acc_train}')
    loss_test, acc_test = test_step(feat_gen, clf, test_loader, optimizer, ep)
    print(f'Ep={ep}, tsloss={loss_test}, tsacc={acc_test}')
