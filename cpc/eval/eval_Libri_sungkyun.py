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
from cpc.eval.utils.file_utils import check_files_exist

DB_WAV_ROOT = './Speaker_ID_DB/Libri-from-sincnet/Librispeech_spkid_sel/'
TR_LIST_PATH = './Speaker_ID_DB/Libri-from-sincnet/libri_tr.scp'
TS_LIST_PATH = './Speaker_ID_DB/Libri-from-sincnet/libri_te.scp'
LABEL_PATH = './Speaker_ID_DB/Libri-from-sincnet/libri_dict.npy'

# Get audio filepaths from .scp files
tr_fps = get_fps_from_txt(TR_LIST_PATH)
ts_fps = get_fps_from_txt(TS_LIST_PATH)
check_files_exist(tr_fps, DB_WAV_ROOT)
check_files_exist(ts_fps, DB_WAV_ROOT)

# Get true labels from .npy files
label_dict = np.load(LABEL_PATH, allow_pickle=True)[()]


"""LibriSel_dataset.py."""
import torch
import random
from torch.utils.data.dataset import Dataset
from torch.multiprocessing import Pool
from pathlib import Path

class LibriSelectionDataset(Dataset):
    """LibriSpeech Selection data from sincnet paper."""

    def __init__(self,
                 win_sz=20480,
                 db_wav_root=DB_WAV_ROOT,
                 fps_list=TS_LIST_PATH,
                 label_path=LABEL_PATH,
                 n_process_loader=4,
                 MAX_SIZE_LOADED=4000000000):
        """
        Args:
            - win_sz (int): size of the sliding window
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
        self.win_sz = win_sz

        """Parsing customized to Libri-selection dataset."""
        fps_name_only = get_fps_from_txt(fps_list)
        self.all_labels_fps = [(label_dict[x], Path(db_wav_root) / Path(x)) for x in fps_name_only]
        assert(check_files_exist(self.all_fps))
        label_dict = np.load(LABEL_PATH, allow_pickle=True)[()]
        self.all_labels = [label_dict[x] for x in fps_name_only]
        
        self.reload_pool = Pool(n_process_loader)
        self.prepare()


    def prepare(self):
        random.shuffle(self.seqNames)
        start_time = time.time()

        print("Checking length...")
        allLength = self.reload_pool.map(extractLength, self.seqNames)

        self.packageIndex, self.totSize = [], 0
        start, packageSize = 0, 0
        for index, length in tqdm.tqdm(enumerate(allLength)):
            packageSize += length
            if packageSize > self.MAX_SIZE_LOADED:
                self.packageIndex.append([start, index])
                self.totSize += packageSize
                start, packageSize = index, 0

        if packageSize > 0:
            self.packageIndex.append([start, len(self.seqNames)])
            self.totSize += packageSize

        print(f"Done, elapsed: {time.time() - start_time:.3f} seconds")
        print(f'Scanned {len(self.seqNames)} sequences '
              f'in {time.time() - start_time:.2f} seconds')
        print(f"{len(self.packageIndex)} chunks computed")
        self.currentPack = -1
        self.nextPack = 0

    
