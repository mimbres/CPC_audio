# Copyright (c) Cochlear.ai, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""".

Created on Fri Mar 27 16:02:58 2020
@author: skchang@cochlear.ai
"""
import numpy as np
from cpc.eval.utils.file_utils import get_fps_from_txt
from cpc.eval.utils.file_utils import check_files_exist_in_dir
from cpc.eval.utils.sampler import UniformAudioSampler, SequentialSampler, SameSpeakerSampler

DB_WAV_ROOT = './Speaker_ID_DB/Libri-from-sincnet/Librispeech_spkid_sel/'
TR_LIST_PATH = './Speaker_ID_DB/Libri-from-sincnet/libri_tr.scp'
TS_LIST_PATH = './Speaker_ID_DB/Libri-from-sincnet/libri_te.scp'
LABEL_PATH = './Speaker_ID_DB/Libri-from-sincnet/libri_dict.npy'

# Get audio filepaths from .scp files
tr_fps = get_fps_from_txt(TR_LIST_PATH)
ts_fps = get_fps_from_txt(TS_LIST_PATH)
check_files_exist_in_dir(tr_fps, DB_WAV_ROOT)
check_files_exist_in_dir(ts_fps, DB_WAV_ROOT)

# Get true labels from .npy files
label_dict = np.load(LABEL_PATH, allow_pickle=True)[()]


"""LibriSel_dataset.py."""
import torch
import random
import time
import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
from torch.multiprocessing import Pool
from pathlib import Path
from cpc.eval.utils.file_utils import extractLength, loadFile

class AudioLoader(object):
    """A DataLoader meant to handle an AudioBatchData object.
    In order to handle big datasets AudioBatchData works with big chunks of
    audio it loads sequentially in memory: once all batches have been sampled
    on a chunk, the AudioBatchData loads the next one.
    """
    def __init__(self,
                 dataset,
                 samplerCall,
                 nLoop,
                 updateCall,
                 size,
                 numWorkers):
        r"""
        Args:
            - dataset (AudioBatchData): target dataset
            - samplerCall (function): batch-sampler to call
            - nLoop (int): number of chunks to load
            - updateCall (function): function loading the next chunk
            - size (int): total number of batches
            - numWorkers (int): see torch.utils.data.DataLoader
        """
        self.samplerCall = samplerCall
        self.updateCall = updateCall
        self.nLoop = nLoop
        self.size = size
        self.dataset = dataset
        self.numWorkers = numWorkers

    def __len__(self):
        return self.size

    def __iter__(self):

        for i in range(self.nLoop):
            sampler = self.samplerCall()
            dataloader = DataLoader(self.dataset,
                                    batch_sampler=sampler,
                                    num_workers=self.numWorkers)
            for x in dataloader:
                yield x
            if i < self.nLoop - 1:
                self.updateCall()



class LibriSelectionDataset(Dataset):
    """LibriSpeech Selection data from sincnet paper."""

    def __init__(self,
                 sizeWindow=20480,
                 db_wav_root=DB_WAV_ROOT,
                 fps_list=str(),
                 label_path=str(),
                 nSpeakers=-1,
                 n_process_loader=50,
                 MAX_SIZE_LOADED=4000000000):
        """Init.
        
        Args:
            - sizeWindow (int): size of the sliding window
            - db_wav_path (str):
            - fps_list_path (str): 
            - label_path (str):
            - n_process_loader (int):
            - MAX_SIZE_LOADED (int): target maximal size of the floating array
                                    containing all loaded data.
                                    
        """
        self.MAX_SIZE_LOADED = MAX_SIZE_LOADED
        self.n_process_loader = n_process_loader
        self.db_wav_root = Path(db_wav_root)
        self.sizeWindow = sizeWindow

        """Parsing customized to Libri-selection dataset."""
        fps_name_only = get_fps_from_txt(fps_list)
        label_dict = np.load(label_path, allow_pickle=True)[()]
        self.all_labels_fps = [(label_dict[x], Path(db_wav_root) / Path(x)) for x in fps_name_only]
        
        self.reload_pool = Pool(n_process_loader)
        self.prepare() # Split large number of files into packages, and set {self.currentPack=-1, self.nextPack=0} 
        
        if nSpeakers==-1:
            nSpeakers = len(set(label_dict.values()))
        self.speakers = list(range(nSpeakers))
        self.data = []
        
        self.loadNextPack(first=True)
        self.loadNextPack()

    def __len__(self):
        """Get length."""
        return self.totSize // self.sizeWindow
        

    def prepare(self):
        """Prepare."""
        random.shuffle(self.all_labels_fps)
        start_time = time.time()

        print("Checking length...")
        allLength = self.reload_pool.map(extractLength, self.all_labels_fps)

        self.packageIndex, self.totSize = [], 0
        start, packageSize = 0, 0
        for index, length in tqdm.tqdm(enumerate(allLength)):
            packageSize += length
            if packageSize > self.MAX_SIZE_LOADED:
                self.packageIndex.append([start, index])
                self.totSize += packageSize
                start, packageSize = index, 0

        if packageSize > 0:
            self.packageIndex.append([start, len(self.all_labels_fps)])
            self.totSize += packageSize

        print(f"Done, elapsed: {time.time() - start_time:.3f} seconds")
        print(f'Scanned {len(self.all_labels_fps)} sequences '
              f'in {time.time() - start_time:.2f} seconds')
        print(f"{len(self.packageIndex)} chunks computed")
        self.currentPack = -1
        self.nextPack = 0

    def clear(self):
        """Clear."""
        if 'data' in self.__dict__:
            del self.data
        if 'speakerLabel' in self.__dict__:
            del self.speakerLabel
        if 'seqLabel' in self.__dict__:
            del self.seqLabel

    def getNPacks(self):
        """Get N packs."""
        return len(self.packageIndex)
    
    def getNSeqs(self):
        """Get N seqs."""
        return len(self.seqLabel) - 1
    
    def getNLoadsPerEpoch(self):
        """Get N loads per epoch."""
        return len(self.packageIndex) 
    
    def getSpeakerLabel(self, idx):
        idSpeaker = next(x[0] for x in enumerate(
            self.speakerLabel) if x[1] > idx) - 1
        return idSpeaker    

    def loadNextPack(self, first=False):
        """Load next pack."""
        self.clear()
        if not first:
            self.currentPack = self.nextPack
            start_time = time.time()
            print('Joining pool')
            self.r.wait()
            print(f'Joined process, elapsed={time.time()-start_time:.3f} secs')
            self.nextData = self.r.get()
            self.parseNextDataBlock()
            del self.nextData
        self.nextPack = (self.currentPack + 1) % len(self.packageIndex)
        seqStart, seqEnd = self.packageIndex[self.nextPack]
        if self.nextPack == 0 and len(self.packageIndex) > 1:
            self.prepare()
        """map() blocks until complete, map_async() returns immediately and 
        schedules a callback to be run on the result."""
        self.r = self.reload_pool.map_async(loadFile,
                                            self.all_labels_fps[seqStart:seqEnd])
        """loadFile: return speaker, seqName, seq"""
        
    def parseNextDataBlock(self):
        """Parse next data block."""
        # Labels
        self.speakerLabel = [0]
        self.seqLabel = [0]
        speakerSize = 0
        indexSpeaker = 0

        # To accelerate the process a bit
        self.nextData.sort(key=lambda x: (x[0], x[1])) 
        """
        nextData[0] = (1243, '4910-14124-0001-1',
                       tensor([-0.0089, -0.0084, -0.0079,  ..., -0.0015, -0.0056,  0.0047]))
        """
        tmpData = []

        for speaker, seqName, seq in self.nextData:
            while self.speakers[indexSpeaker] < speaker:
                indexSpeaker += 1
                self.speakerLabel.append(speakerSize)
            if self.speakers[indexSpeaker] != speaker:
                raise ValueError(f'{speaker} invalid speaker')

            sizeSeq = seq.size(0)
            tmpData.append(seq)
            self.seqLabel.append(self.seqLabel[-1] + sizeSeq)
            speakerSize += sizeSeq
            del seq

        self.speakerLabel.append(speakerSize)
        self.data = torch.cat(tmpData, dim=0)

    def __getitem__(self, idx):
        """Get item."""
        if idx < 0 or idx >= len(self.data) - self.sizeWindow - 1:
            print(idx)

        outData = self.data[idx:(self.sizeWindow + idx)].view(1, -1)
        label = torch.tensor(self.getSpeakerLabel(idx), dtype=torch.long)

        return outData, label
    
    def getBaseSampler(self, type, batchSize, offset):
        """Get base sampler."""
        if type == "samespeaker":
            return SameSpeakerSampler(batchSize, self.speakerLabel,
                                      self.sizeWindow, offset)
        if type == "samesequence":
            return SameSpeakerSampler(batchSize, self.seqLabel,
                                      self.sizeWindow, offset)
        if type == "sequential":
            return SequentialSampler(len(self.data), self.sizeWindow,
                                     offset, batchSize)
        sampler = UniformAudioSampler(len(self.data), self.sizeWindow,
                                      offset)
        return BatchSampler(sampler, batchSize, True)

    def getDataLoader(self, batchSize, type, randomOffset, numWorkers=0,
                      onLoop=-1):
        """Get a batch sampler for the current dataset.
        
        Args:
            - batchSize (int): batch size
            - groupSize (int): in the case of type in ["samespeaker", "samesequence"]
            number of items sharing a same label in the group
            (see AudioBatchSampler)
            - type (string):
                type == "samespeaker": grouped sampler speaker-wise
                type == "samesequence": grouped sampler sequence-wise
                type == "sequential": sequential sampling
                else: uniform random sampling of the full audio
                vector
            - randomOffset (bool): if True add a random offset to the sampler
                                   at the begining of each iteration
                                   
        """
        nLoops = len(self.packageIndex)
        totSize = self.totSize // (self.sizeWindow * batchSize)
        if onLoop >= 0:
            self.currentPack = onLoop - 1
            self.loadNextPack()
            nLoops = 1

        def samplerCall():
            offset = random.randint(0, self.sizeWindow // 2) \
                if randomOffset else 0
            return self.getBaseSampler(type, batchSize, offset)

        return AudioLoader(self, samplerCall, nLoops, self.loadNextPack,
                           totSize, numWorkers)


#%%
def unit_test():
    #[key for key in label_dict if (label_dict[key] == 355)]
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    db_test = LibriSelectionDataset(sizeWindow=20480*5, db_wav_root=DB_WAV_ROOT, 
                                     fps_list=TS_LIST_PATH, label_path=LABEL_PATH,
                                     n_process_loader=2, MAX_SIZE_LOADED=40000000)
    
    db_train = LibriSelectionDataset(sizeWindow=20480, db_wav_root=DB_WAV_ROOT, 
                                     fps_list=TR_LIST_PATH, label_path=LABEL_PATH,
                                     n_process_loader=4, MAX_SIZE_LOADED=400000000)
    
    # type = ['samespeaker', 'samesequence', 'sequential', 'uniform']
    test_loader = db_test.getDataLoader(batchSize=8, type='samespeaker',
                                        randomOffset=False, numWorkers=0)
    
    train_loader = db_train.getDataLoader(batchSize=64, type='sequential',
                                         randomOffset=False, numWorkers=0)
    
    
    for step, (batch_data, label) in enumerate(train_loader):
        #if (len(label) < 2): print(label)
        if (123 in label): break;
    print(label)
        
    for step, (batch_data, label) in enumerate(test_loader):
        if label[0]==123: break;

    

