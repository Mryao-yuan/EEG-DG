#!/usr/bin/env python
# coding: utf-8


from torch.utils.data import Dataset
import os
import pickle
import csv


class eegDataset(Dataset):

    def __init__(self, dataPath, dataLabelsPath, transform=None, preloadData=False):

        self.data = []
        self.dataPath = dataPath
        self.labels = []
        self.dataLabelsPath = dataLabelsPath
        self.preloadData = preloadData
        self.transform = transform

        # Load the labels file
        # id,relativeFilePath,label,subject,session
        with open(self.dataLabelsPath, "r") as f:
            eegReader = csv.reader(f, delimiter=',')
            for row in eegReader:
                self.labels.append(row)

            # remove the first header row
            del self.labels[0] # id,file_name,label,subjetc, session

        # convert the labels to int
        for i, label in enumerate(self.labels):
            self.labels[i][2] = int(self.labels[i][2])

        # if preload data is true then load all the data and apply the transforms as well
        if self.preloadData:
            for i, trial in enumerate(self.labels):
                with open(os.path.join(self.dataPath, trial[1]), 'rb') as fp:
                    d = pickle.load(fp)
                    if self.transform:
                        d = self.transform(d)
                    self.data.append(d)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        '''Load and provide the data and label'''

        if self.preloadData:
            data = self.data[idx]

        else:
            with open(os.path.join(self.dataPath, self.labels[idx][1]), 'rb') as fp:
                data = pickle.load(fp)
                if self.transform:
                    data = self.transform(data)
        # data,label
        d = [data[0], data[1]]
        return d

    def createPartialDataset(self, idx, loadNonLoadedData=False):
        # 基于 train/val inddex 修改对应顺序
        self.labels = [self.labels[i] for i in idx]

        if self.preloadData:
            self.data = [self.data[i] for i in idx]
        elif loadNonLoadedData:
            for i, trial in enumerate(self.labels):
                with open(os.path.join(self.dataPath, trial[1]), 'rb') as fp:
                    d = pickle.load(fp)
                    if self.transform:
                        d = self.transform(d)
                    self.data.append(d)
            self.preloadData = True





