#!/usr/bin/env python
# coding: utf-8

# To do deep learning
import os
import sys
import copy
import time
import pickle
import random
import gc

import higher
import numpy as np
import torch
import collections
from torch import cat
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import torch.utils.data.sampler as builtInSampler
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, accuracy_score

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

Path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, Path)
from utils import stopCriteria, samplers
from utils.tools import RandomDomainSampler, random_pairs_of_minibatches, random_split, ForeverDataIterator
from lossFunction import coral, scl, irm, mmd, mcc
from utils.NeuronCLustering import NeuronClustering, get_norm


class baseModel:
    def __init__(
            self,
            net,
            network,
            resultsSavePath=None,
            seed=961102,
            setRng=True,
            preferedDevice='gpu',
            nGPU=0,
            batchSize=1,
            tradeOff=0.,
            tradeOff2=0.,
            tradeOff3=0.,
            tradeOff4=0,
            ndomain=9,
            classes=4,
            algorithm='ce',
            coco_config=None):
        self.network = network
        self.seed = seed
        self.preferedDevice = preferedDevice
        self.batchSize = batchSize
        self.setRng = setRng
        self.resultsSavePath = resultsSavePath
        self.device = None
        self.ndomain = ndomain - 1

        self.tradeOff = tradeOff
        self.tradeOff2 = tradeOff2
        self.tradeOff3 = tradeOff3
        self.tradeOff4 = tradeOff4
        # Set RNG
        if self.setRng:
            self.setRandom(self.seed)

        # Set device
        self.setDevice(nGPU)
        self.net = net.to(self.device)
        self.nclasses = classes
        self.classes = [i for i in range(self.nclasses)]
        self.coral = coral.CorrelationAlignmentLoss().to(self.device)
        self.scl = scl.SupConLoss().to(self.device)
        self.irm = irm.InvariancePenaltyLoss().to(self.device)
        self.mmd = mmd.MMDLoss().to(self.device)
        self.mse = nn.MSELoss().to(self.device)

        self.mcc = mcc.MinimumClassConfusionLoss(temperature=2.5).to(self.device)

        # ['ce', 'coral', 'scl', 'smcldgn', 'smcldgn_mc', 'irm', 'mixup', 'mmd', 'mldg','coco']
        self.trainOneEpoch = vars(baseModel)[algorithm]

        if self.resultsSavePath is not None:
            if not os.path.exists(self.resultsSavePath):
                os.makedirs(self.resultsSavePath)
            print('Results will be saved in folder : ' + self.resultsSavePath)
        # config coco 
        # --- init CoCo 聚类器 (NACT) ---
        self.use_coco = (algorithm in ['coco', 'coco_scl', 'coco_condcad']) or (coco_config is not None)
        self.c_clusters = None 
        self.c_weights = None 
        self.unact_neurons = None
        if self.use_coco:
            default_coco_config = {
                'label_ids': range(self.nclasses),
                'reg_layer_list': ['feature'], # baseon model name
                'quantile': 0.05, # for threshold of activation
                # 'sample_ratio':0.5,
                'act_ratio': 0.05, # 激活比例阈值，< act_ratio 视为噪声，丢弃
                'num_concept_clusters': 5,
                'clustering_method': 'kmeans',
                'low_level_merge_method': 'jaccard',
                'high_level_merge_method': 'jaccard',
                'domain_level': False, #  DG 任务通常为 False
                'class_level': True,
                'multi_level': True
            }
            if coco_config:
                default_coco_config.update(coco_config)
            
            self.NACT = NeuronClustering(**default_coco_config)
            self.coco_projection = nn.Sequential(
                nn.Linear(320, 128), # for bci42a
                nn.ReLU(),
                nn.Linear(128, 128)
            ).to(self.device)
        
        
    def train(
            self,
            trainData,
            valData,
            testData=None,
            optimFns='Adam',
            optimFnArgs={},
            sampler=None,
            lr=0.001,
            stopCondi=None,
            loadBestModel=True,
            bestVarToCheck='valInacc'):

        # define the classes
        classes = [i for i in range(self.nclasses)]

        # Define the sampler
        if sampler is not None:
            # sampler = self._findSampler(sampler)
            sampler = RandomDomainSampler(trainData, self.batchSize, n_domains_per_batch=self.ndomain)
        # Create the loss function
        lossFn = self._findLossFn('CrossEntropyLoss')(reduction='sum')
        # lossFn = nn.CrossEntropyLoss().cuda()
        # store the experiment details.
        self.expDetails = []

        # Lets run the experiment
        expNo = 0
        original_net_dict = copy.deepcopy(self.net.state_dict())
        # set the details
        expDetail = {'expNo': expNo, 'expParam': {'optimFn': optimFns,
                                                  'lossFn': lossFn, 'lr': lr,
                                                  'stopCondi': stopCondi}}
        # Reset the network to its initial form.
        self.net.load_state_dict(original_net_dict)

        # Run the training and get the losses.
        trainResults = self._trainOE(trainData, valData,testData, optimFns, lr, stopCondi,
                                     optimFnArgs, classes=classes,
                                     sampler=sampler,
                                     loadBestModel=loadBestModel,
                                     bestVarToCheck=bestVarToCheck)

        # store the results and netParm
        expDetail['results'] = {'train': trainResults}
        expDetail['netParam'] = copy.deepcopy(self.net.to('cpu').state_dict())

        self.net.to(self.device)
        # If you are restoring the best model at the end of training then get the final results again.
        pred, act, l = self.predict(trainData, sampler=None)
        trainResultsBest = self.calculateResults(pred, act, classes=classes)
        trainResultsBest['loss'] = l
        pred, act, l = self.predict(valData, sampler=None)
        valResultsBest = self.calculateResults(pred, act, classes=classes)
        valResultsBest['loss'] = l
        expDetail['results']['trainBest'] = trainResultsBest
        expDetail['results']['valBest'] = valResultsBest

        # if test data is present then get the results for the test data.
        if testData is not None:
            pred, act, l = self.predict(testData, sampler=None)
            testResults = self.calculateResults(pred, act, classes=classes)
            testResults['loss'] = l
            expDetail['results']['test'] = testResults

        # Print the final output to the console:
        print("Exp No. : " + str(expNo + 1))
        print('________________________________________________')
        print("\n Train Results: ")
        print(expDetail['results']['trainBest'])
        print('\n Validation Results: ')
        print(expDetail['results']['valBest'])
        if testData is not None:
            print('\n Test Results: ')
            print(expDetail['results']['test'])

        # save the results
        if self.resultsSavePath is not None:
            # Store the graphs
            self.plotLoss(trainResults['trainLoss'], trainResults['valLoss'],
                          savePath=os.path.join(self.resultsSavePath,
                                                'exp-' + str(expNo) + '-loss.png'))
            self.plotAcc(trainResults['trainResults']['acc'],
                         trainResults['valResults']['acc'],
                         trainResults['testResults']['acc'],
                         savePath=os.path.join(self.resultsSavePath,
                                               'exp-' + str(expNo) + '-acc.png'))

            # Store the data in experimental details.
            with open(os.path.join(self.resultsSavePath, 'expResults' +
                                                         str(expNo) + '.dat'), 'wb') as fp:
                pickle.dump(expDetail, fp)
            # Store the net parameters in experimental details.
            # https://zhuanlan.zhihu.com/p/129948825 store model methods.
            model_path = self.resultsSavePath + '\\checkpoint.pth.tar'
            torch.save({'state_dict': self.net.state_dict()}, model_path)

        # Increment the expNo
        self.expDetails.append(expDetail)
        expNo += 1

    def _trainOE(
            self,
            trainData,
            valData,
            testData,
            optimFn='Adam',
            lr=0.001,
            stopCondi=None,
            optimFnArgs={},
            loadBestModel=True,
            bestVarToCheck='valLoss',
            classes=None,
            sampler=None, ):

        # For reporting.
        trainResults = []
        valResults = []
        testResults = []
        trainLoss = []
        valLoss = []
        loss = []
        bestNet = copy.deepcopy(self.net.state_dict())
        bestValue = float('inf')
        earlyStopReached = False
        bestEpoch = 0

        # Create optimizer
        self.optimizer = self._findOptimizer(optimFn)(self.net.parameters(), lr=lr, **optimFnArgs)
        bestOptimizerState = copy.deepcopy(self.optimizer.state_dict())

        # Initialize the stop criteria
        stopCondition = stopCriteria.composeStopCriteria(**stopCondi)

        # lets start the training.
        monitors = {'epoch': 0, 'valLoss': 10000, 'valInacc': 1}
        doStop = False
        while not doStop:
            if self.use_coco:
                current_epoch = monitors['epoch']
                cluster_schedule = [i for i in range(5,200,1)]
                # cluster in all epochs
                if current_epoch < cluster_schedule[0]:
                    self.tradeOff2 = 0.0 # 强制关闭 CoCo Loss
                    if current_epoch == 0: 
                        print(f">> Warm-up: CoCo disabled for first {cluster_schedule[0]} epochs.")
                else:
                    self.tradeOff2 =  0.1
                    if current_epoch in cluster_schedule:
                        print(f">> Warm-up ended. CoCo enabled with weight {self.tradeOff2}.")
                        self.set_clusters(trainData)
            # train the epoch.
            start = time.time()
            loss.append(self.trainOneEpoch(self, trainData, self.optimizer, sampler=sampler))

            # evaluate the training and validation accuracy.
            pred, act, l = self.predict(trainData, sampler=None)
            trainResults.append(self.calculateResults(pred, act, classes=classes))
            trainLoss.append(l)
            monitors['trainLoss'] = l
            monitors['trainInacc'] = 1 - trainResults[-1]['acc']

            pred, act, l = self.predict(valData, sampler=None)
            valResults.append(self.calculateResults(pred, act, classes=classes))
            valLoss.append(l)
            monitors['valLoss'] = l
            monitors['valInacc'] = 1 - valResults[-1]['acc']

            # test
            if testData is not None:
                pred, act, l = self.predict(testData, sampler=None)
                testResults.append(self.calculateResults(pred, act, classes=classes))
                monitors['testLoss'] = l
                monitors['testInacc'] = 1 - testResults[-1]['acc']
            runTime = time.time() - start
            # print the epoch info
            print("\t \t Epoch " + str(monitors['epoch'] + 1))
            print("Train loss = " + "%.3f" % trainLoss[-1] + " Train Acc = " +
                  "%.3f" % trainResults[-1]['acc'] +
                  ' Val Acc = ' + "%.3f" % valResults[-1]['acc'] +
                  ' Val loss = ' + "%.3f" % valLoss[-1] +
                  ' Epoch Time = ' + "%.3f" % runTime)

            if loadBestModel:
                if monitors[bestVarToCheck] < bestValue:
                    bestValue = monitors[bestVarToCheck]
                    bestNet = copy.deepcopy(self.net.state_dict())
                    bestOptimizerState = copy.deepcopy(self.optimizer.state_dict())
                    bestEpoch = monitors['epoch']
            # Check if to stop
            doStop = stopCondition(monitors)

            # Check if we want to continue the training after the first stop:
            if doStop:
                # first load the best model
                if loadBestModel and not earlyStopReached:
                    self.net.load_state_dict(bestNet)
                    self.optimizer.load_state_dict(bestOptimizerState)

                # Now check if  we should continue training:
            # update the epoch
            monitors['epoch'] += 1

        # Make individual list for components of trainResults and valResults
        t = {}
        v = {}
        test = {}
        for key in trainResults[0].keys():
            t[key] = [result[key] for result in trainResults]
            v[key] = [result[key] for result in valResults]
            test[key] = [result[key] for result in testResults]
            

        return {'trainResults': t, 'valResults': v, 'testResults': test,
                'trainLoss': trainLoss, 'valLoss': valLoss, 'bestEpoch': bestEpoch, }

    def extract_features_for_clustering(self, dataset, max_samples=1024):
        self.net.eval()
        n_samples = len(dataset)
        if n_samples <= max_samples:
            n_subset = n_samples
            subset = dataset 
        else:
            n_subset = max_samples
            indices = np.random.choice(n_samples, n_subset, replace=False)
            subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=128, shuffle=False, num_workers=2)
        
        d_layer_output = []
        
        with torch.no_grad():
            for d in loader:
                x = d[0].to(self.device).permute(0, 3, 1, 2)
                y = d[1].type(torch.LongTensor).to(self.device)
                _, feats, _= self.net.update(x) # logits, features, projections
                
                feat_to_save = feats[-1] if isinstance(feats, (list, tuple)) else  feats
                feat_to_save = get_norm(feat_to_save)
                

                d_layer_output.append({
                    "feature": feat_to_save.cpu(), # 键名与 reg_layer_list 一致
                    "labels": y.cpu(),
                    "accs": None 
                })
        self.net.train()
        return d_layer_output

    def set_clusters(self, trainData, update=True):
        # 1. 提取特征 (建议将采样数调大以接近源码的丰富度)
        # 源码是在 evaluation 阶段收集全量特征，这里用采样模拟
        d_layer_output = self.extract_features_for_clustering(trainData, max_samples=4096000)
        # 2. 获取配置参数 (模拟源码的 hparams)
        # 如果你的 self 中没有这些属性，这里给了默认值
        include_unact = getattr(self, 'include_unact', False) # 是否包含未激活神经元
        use_c_weights = getattr(self, 'use_c_weights', True)  # 是否使用权重
        
        # 3. 调用 NACT 计算聚类
        self.NACT.compute_neuron_cluster(d_layer_output, padding=True, unact=include_unact)
        
        # 4. 更新模型状态 
        if update:
            self.c_clusters = {
                l: torch.stack(v[0], dim=0).to(self.device) 
                for l, v in self.NACT.layer_act_neuron_dict.items()
            }
            if use_c_weights:
                self.c_weights = {
                    l: torch.stack(v[1], dim=0).to(self.device) 
                    for l, v in self.NACT.layer_act_neuron_dict.items()
                }
            
            # [Source Style] 处理未激活神经元 (如果你未来开启的话)
            if include_unact:
                self.unact_neurons = {
                    l: v.to(self.device) 
                    for l, v in self.NACT.layer_unact_neurons.items()
                }
            if not hasattr(self, 'c_size'):
                self.c_size = {}
            
            self.c_size['feature'] = 320
            if 'feature' in self.c_clusters:
                print(f">> Clusters updated. Shape: {self.c_clusters['feature'].shape}")

        # 5. 内存清理 (保持你的好习惯，源码在外部清理，这里内部清理更安全)
        del d_layer_output
        import gc
        gc.collect()
    
    def abstract_concept_from_feat(self, feat, all_y=None, l_name='feature'):
        """
        从原始特征 z 中提取概念向量 v
        """
        #  feat: batch_size x feature_dim 
        b = feat.shape[0]
        if feat.dim() > 2:
            feat = feat.view(feat.size(0), -1)
        
        # [num_cluster, c_neurons]
        # Compute concept-level contrastive loss
        # [batch, num_cluster, c_neurons]
        cluster_n_ids = self.c_clusters[l_name].to(self.device).expand(b, -1, -1)  # b, num_cluster, neurons
        
        # 假设 padding_idx 是 feature_dim (即最后一列)
        # Add padding neuron to the layer
        padding = torch.zeros((b, 1), requires_grad=False, device=self.device)
        feat = torch.cat([feat, padding], dim=1) # [B, N+1]

        # 4. 处理索引越界
        # 源码中 padding 的 index 可能是 -1 或 feature_dim，我们需要将其统一指向最后一列 (feature_dim)
        # Expand tensor for gathering the concept neurons
        # [batch, num_cluster, neurons] -> [batch, num_cluster, c_neurons]
        num_concepts = cluster_n_ids.shape[1]
        expanded_feat = feat.unsqueeze(1).expand(-1, num_concepts, -1) # b, num_concepts, N+1
        concept_state = expanded_feat.gather(-1, index=cluster_n_ids) 

        # 6. 加权求和
        if self.c_weights is None:
            # 如果没有权重，取最大值或平均值
            concept_state_vec =  concept_state.amax(-1) # 返回 concept在每个簇上的最大激活值
        else:
            c_weights = self.c_weights[l_name].to(self.device)
            # 权重同样需要处理 padding，不过通常权重对应位置为0，直接乘即可
            c_weights_expanded = c_weights.unsqueeze(0).expand(b, -1, -1)
            concept_state_vec = (concept_state * c_weights_expanded).sum(dim=-1)

        if self.unact_neurons is not None:
            unact_nid = self.unact_neurons[l_name]
            unact_states = torch.index_select(feat, dim=1, index=unact_nid)  # [b, m]
            concept_state_vec = torch.cat([concept_state_vec, unact_states], dim=1)
            # print(unact_states.shape, concept_state_vec.shape, unact_states)

        padding = torch.zeros((b, self.c_size[l_name] - concept_state_vec.shape[-1]),
                              requires_grad=False, device='cuda')
        concept_state_vec = torch.cat([concept_state_vec, padding], dim=1)
        return concept_state_vec
    
    # methods 
    def ce(self, trainData, optimizer, sampler=None, **kwargs):

        self.net.train()
        running_loss = 0
        dataLoader = DataLoader(trainData, batch_size=self.batchSize, sampler=sampler, drop_last=True)
        for d in dataLoader:
            with torch.enable_grad():
                optimizer.zero_grad()

                x = d[0].to(self.device).permute(0, 3, 1, 2)
                labels = d[1].type(torch.LongTensor).to(self.device)
                logits, _, _ = self.net.update(x)

                loss = F.cross_entropy(logits, labels)
                # backward pass
                loss.backward()
                optimizer.step()
            # accumulate the loss over mini-batches.
            running_loss += loss.data

        # return the present lass. This may be helpful to stop or continue the training.
        return running_loss.item() / len(dataLoader)

    def scl(self, trainData, optimizer, sampler=None, **kwargs):
        self.net.train()
        running_loss = 0
        dataLoader = DataLoader(trainData, batch_size=self.batchSize, sampler=sampler, drop_last=True)
        for d in dataLoader:
            with torch.enable_grad():

                optimizer.zero_grad()
                all_x = d[0].to(self.device).permute(0, 3, 1, 2)
                all_y = d[1].type(torch.LongTensor).to(self.device)
                # subject idx
                # all_d = d[2].type(torch.LongTensor).to(self.device)
                batch_size = all_y.size()[0]
                lam = np.random.uniform(0.9, 1.0)

                with torch.no_grad():
                    sorted_y, indices = torch.sort(all_y)
                    sorted_x = torch.zeros_like(all_x)
                    # sorted_d = torch.zeros_like(all_d)
                    for idx, order in enumerate(indices):
                        sorted_x[idx] = all_x[order]
                        # sorted_d[idx] = all_d[order]
                    intervals = []
                    ex = 0
                    for idx, val in enumerate(sorted_y):
                        if ex == val:
                            continue
                        intervals.append(idx)
                        ex = val
                    intervals.append(batch_size)

                    all_x = sorted_x
                    all_y = sorted_y
                    # all_d = sorted_d
                
                all_y =  all_y if 'EEGSimpleConv' not in self.network else torch.repeat_interleave(all_y, repeats=all_x.shape[1], dim=0)
                
                output, _, proj = self.net.update(all_x)

                loss_ce = F.cross_entropy(output, all_y)
                # shuffle
                mix1 = torch.zeros_like(proj)
                mix2 = torch.zeros_like(proj)
                ex = 0
                for end in intervals:
                    shuffle_indices = torch.randperm(end - ex) + ex
                    shuffle_indices2 = torch.randperm(end - ex) + ex
                    for idx in range(end - ex):
                        mix1[idx + ex] = proj[shuffle_indices[idx]]
                        mix2[idx + ex] = proj[shuffle_indices2[idx]]
                    ex = end

                p1 = lam * proj + (1 - lam) * mix1
                p2 = lam * proj + (1 - lam) * mix2
                # combine multiple views
                p = torch.cat([p1.unsqueeze(1), p2.unsqueeze(1)], dim=1)
                scl_loss = self.scl(p, all_y, mask=None, d=None)

                loss = loss_ce + scl_loss * self.tradeOff

                # backward pass
                loss.backward()
                optimizer.step()
            # accumulate the loss over mini-batches.
            running_loss += loss.data

        # return the present lass. This may be helpful to stop or continue the training.
        return running_loss.item() / len(dataLoader)

    def coco_scl(self, trainData, optimizer, sampler=None, **kwargs):

        self.net.train()
        running_loss = 0
        dataLoader = DataLoader(trainData, batch_size=self.batchSize, sampler=sampler, drop_last=True)
        for d in dataLoader:
            with torch.enable_grad():
                optimizer.zero_grad()
                all_x = d[0].to(self.device).permute(0, 3, 1, 2)
                all_y = d[1].type(torch.LongTensor).to(self.device)
                
                
                with torch.no_grad():
                    sorted_y, indices = torch.sort(all_y)
                    sorted_x = torch.zeros_like(all_x)
                    for idx, order in enumerate(indices):
                        sorted_x[idx] = all_x[order]
                    intervals = []
                    ex = 0
                    for idx, val in enumerate(sorted_y):
                        if ex == val:
                            continue
                        intervals.append(idx)
                        ex = val
                    intervals.append(all_y.size()[0])

                    all_x = sorted_x
                    all_y = sorted_y
                

                output, feats, _ = self.net.update(all_x)
               
                loss_ce = F.cross_entropy(output, all_y)
                
                # coco loss
                loss_coco = 0.0
                if self.tradeOff2 > 0 and self.c_clusters is not None:
                    z = feats[-1] if isinstance(feats, (list, tuple)) else feats
                    
                    # 1. 提取概念特征 v [Batch, K]
                    # 调用你之前定义好的辅助函数 (确保使用了 padding 修复版)
                    feat_c = self.abstract_concept_from_feat(z, sorted_y) # 还需要改进，在没有cluster的时候
                    proj = self.coco_projection(feat_c)
                    proj = F.normalize(proj, dim=1)
                    # 2. 准备 Shuffle 容器
                    output_2 = torch.zeros_like(output) # 用于同类对齐 Logits
                    output_3 = torch.zeros_like(output) 
                    feat_2 = torch.zeros_like(proj) # 用于同类对齐 features
                    feat_3 = torch.zeros_like(proj)
                    
                    lam = np.random.beta(0.5, 0.5)
                    ex = 0
                    for end in intervals:
                        # 在当前类别的区间内进行打乱 (Intra-class Shuffle)
                        num_samples = end - ex
                        shuffle_indices = torch.randperm(num_samples) + ex
                        shuffle_indices2 = torch.randperm(num_samples) + ex
                        for idx in range(end - ex):
                            output_2[idx + ex] = output[shuffle_indices[idx]]
                            feat_2[idx + ex] = proj[shuffle_indices[idx]]
                            output_3[idx + ex] = output[shuffle_indices2[idx]]
                            feat_3[idx + ex] = proj[shuffle_indices2[idx]]
                        ex = end
                        
                    # 3. Mixup 
                    output_3 = lam * output_2 + (1 - lam) * output_3
                    feat_3 = lam * feat_2 + (1 - lam) * feat_3

                    # regularization in same class
                    L_ind_logit = self.mse(output, output_2)
                    L_hdl_logit = self.mse(output, output_3)
                    L_ind_feat = 0.3 * self.mse(proj, feat_2) # 0.3
                    L_hdl_feat = 0.3 * self.mse(proj, feat_3)
                    # 总 CoCo Loss
                    loss_coco =  (lam * (L_ind_logit + L_ind_feat) + (1 - lam) * (L_hdl_logit + L_hdl_feat))

            # C_scale: 调整权重，防止 CoCo Loss 过大
            C_scale = min(loss_ce.item(), 1.0) 
            loss = loss_ce + C_scale * self.tradeOff2 * loss_coco
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
            optimizer.step()
            running_loss += loss.item()

        return running_loss / len(dataLoader)


    def smcldgn(self, trainData, optimizer, sampler=None, **kwargs):

        self.net.train()
        running_loss = 0
        dataLoader = DataLoader(trainData, batch_size=self.batchSize, sampler=sampler, drop_last=True)
        for d in dataLoader:
            with torch.enable_grad():

                optimizer.zero_grad()
                # b,c,times,subjects ->b,subejct,c,times
                x = d[0].to(self.device).permute(0, 3, 1, 2)
                y = d[1].type(torch.LongTensor).to(self.device)
                batch_size = y.size()[0]

                y_logit, feats, proj = self.net.update(x)
                # based on domains num split, 在数据加载的过程中，是不是按照域的顺序，例如batch 32 则，每个域都加载对应的数据了？？
                y_logit = y_logit.chunk(self.ndomain, dim=0)
                feats = feats.chunk(self.ndomain, dim=0)
                y_coral = y.chunk(self.ndomain, dim=0)

                # coral
                loss_ce = 0
                loss_penalty = 0
                for domain_i in range(self.ndomain):
                    # cls loss
                    y_i, labels_i = y_logit[domain_i], y_coral[domain_i]
                    loss_ce += F.cross_entropy(y_i, labels_i) # cls loss
                    # correlation alignment loss
                    for domain_j in range(domain_i + 1, self.ndomain):
                        f_i = feats[domain_i]
                        f_j = feats[domain_j]
                        loss_penalty += self.coral(f_i, f_j) # adjacent domain

                loss_ce /= self.ndomain
                loss_penalty /= self.ndomain * (self.ndomain - 1) / 2 # domain alignment loss

                # scl
                lam = np.random.uniform(0.9, 1.0)

                sorted_y, indices = torch.sort(y)
                sorted_proj = torch.zeros_like(proj)
                for idx, order in enumerate(indices):
                    sorted_proj[idx] = proj[order]
                intervals = []
                ex = 0
                for idx, val in enumerate(sorted_y):
                    if ex == val:
                        continue
                    intervals.append(idx)
                    ex = val
                intervals.append(batch_size)

                proj = sorted_proj
                y_scl = sorted_y

                # shuffle
                mix1 = torch.zeros_like(proj)
                mix2 = torch.zeros_like(proj)
                ex = 0
                for end in intervals:
                    shuffle_indices = torch.randperm(end - ex) + ex
                    shuffle_indices2 = torch.randperm(end - ex) + ex
                    for idx in range(end - ex):
                        mix1[idx + ex] = proj[shuffle_indices[idx]]
                        mix2[idx + ex] = proj[shuffle_indices2[idx]]
                    ex = end

                p1 = lam * proj + (1 - lam) * mix1
                p2 = lam * proj + (1 - lam) * mix2

                p = torch.cat([p1.unsqueeze(1), p2.unsqueeze(1)], dim=1)
                scl_loss = self.scl(p, y_scl, mask=None)

                loss = loss_ce + self.tradeOff * loss_penalty + self.tradeOff2 * scl_loss

                # backward pass
                loss.backward()
                optimizer.step()
            # accumulate the loss over mini-batches.
            running_loss += loss.data

        # return the present lass. This may be helpful to stop or continue the training.
        return running_loss.item() / len(dataLoader)

    def smcldgn_mc(self, trainData, optimizer, sampler=None, **kwargs):

        self.net.train()
        running_loss = 0
        dataLoader = DataLoader(trainData, batch_size=self.batchSize, sampler=sampler, drop_last=True, )
        for d in dataLoader:
            with torch.enable_grad():

                optimizer.zero_grad()

                x = d[0].to(self.device).permute(0, 3, 1, 2)
                y = d[1].type(torch.LongTensor).to(self.device)
                batch_size = y.size()[0]

                y_logit, feats, proj = self.net.update(x)

                # cls
                loss_ce = F.cross_entropy(y_logit, y)

                feats = feats.chunk(self.ndomain, dim=0)

                # coral
                # select n domains
                lis = list(range(self.ndomain))
                slice = random.sample(lis, self.tradeOff4)
                loss_penalty = 0
                for domain_i in range(len(slice)):
                    # correlation alignment loss
                    for domain_j in range(domain_i + 1, len(slice)):
                        f_i = feats[slice[domain_i]]
                        f_j = feats[slice[domain_j]]
                        loss_penalty += self.coral(f_i, f_j)

                loss_penalty /= len(slice) * (len(slice) - 1) / 2

                # scl
                lam = np.random.uniform(0.9, 1.0)

                sorted_y, indices = torch.sort(y)
                sorted_proj = torch.zeros_like(proj)
                for idx, order in enumerate(indices):
                    sorted_proj[idx] = proj[order]
                intervals = [] # log class intervals
                ex = 0
                for idx, val in enumerate(sorted_y):
                    if ex == val:
                        continue
                    intervals.append(idx)
                    ex = val
                intervals.append(batch_size)

                proj = sorted_proj
                y_scl = sorted_y

                # shuffle
                mix1 = torch.zeros_like(proj)
                mix2 = torch.zeros_like(proj)
                ex = 0
                for end in intervals:
                    shuffle_indices = torch.randperm(end - ex) + ex
                    shuffle_indices2 = torch.randperm(end - ex) + ex
                    for idx in range(end - ex):
                        mix1[idx + ex] = proj[shuffle_indices[idx]]
                        mix2[idx + ex] = proj[shuffle_indices2[idx]]
                    ex = end

                p1 = lam * proj + (1 - lam) * mix1
                p2 = lam * proj + (1 - lam) * mix2

                p = torch.cat([p1.unsqueeze(1), p2.unsqueeze(1)], dim=1)
                scl_loss = self.scl(p, y_scl, mask=None)

                loss = loss_ce + self.tradeOff * loss_penalty + self.tradeOff2 * scl_loss

                loss.backward()
                optimizer.step()
            # accumulate the loss over mini-batches.
            running_loss += loss.data

        # return the present lass. This may be helpful to stop or continue the training.
        return running_loss.item() / len(dataLoader)

    def coral(self, trainData, optimizer, sampler=None, **kwargs):

        self.net.train()
        running_loss = 0
        dataLoader = DataLoader(trainData, batch_size=self.batchSize, sampler=sampler, drop_last=True)

        for d in dataLoader:
            with torch.enable_grad():
                # zero the parameter gradients
                optimizer.zero_grad()

                x = d[0].to(self.device).permute(0, 3, 1, 2)
                labels = d[1].type(torch.LongTensor).to(self.device)

                y_all, feats, _ = self.net.update(x)

                y_all = y_all.chunk(self.ndomain, dim=0)
                feats = feats.chunk(self.ndomain, dim=0)
                labels = labels.chunk(self.ndomain, dim=0)

                loss_ce = 0
                loss_penalty = 0
                for domain_i in range(self.ndomain):
                    # cls loss
                    y_i, labels_i = y_all[domain_i], labels[domain_i]
                    loss_ce += F.cross_entropy(y_i, labels_i)
                    # correlation alignment loss
                    for domain_j in range(domain_i + 1, self.ndomain):
                        f_i = feats[domain_i]
                        f_j = feats[domain_j]
                        loss_penalty += self.coral(f_i, f_j)

                loss_ce /= self.ndomain
                loss_penalty /= self.ndomain * (self.ndomain - 1) / 2

                loss = loss_ce + self.tradeOff * loss_penalty

                # backward pass
                loss.backward()
                optimizer.step()
            # accumulate the loss over mini-batches.
            running_loss += loss.data

        return running_loss.item() / len(dataLoader)

    def mmd(self, trainData, optimizer, sampler=None, **kwargs):

        self.net.train()
        running_loss = 0
        dataLoader = DataLoader(trainData, batch_size=self.batchSize, sampler=sampler, drop_last=True)

        for d in dataLoader:
            with torch.enable_grad():
                # zero the parameter gradients
                optimizer.zero_grad()

                x = d[0].to(self.device).permute(0, 3, 1, 2)
                labels = d[1].type(torch.LongTensor).to(self.device)

                y_all, feats, _ = self.net.update(x)

                y_all = y_all.chunk(self.ndomain, dim=0)
                feats = feats.chunk(self.ndomain, dim=0)
                labels = labels.chunk(self.ndomain, dim=0)

                loss_ce = 0
                loss_penalty = 0
                for domain_i in range(self.ndomain):
                    # cls loss
                    y_i, labels_i = y_all[domain_i], labels[domain_i]
                    loss_ce += F.cross_entropy(y_i, labels_i)
                    # correlation alignment loss
                    for domain_j in range(domain_i + 1, self.ndomain):
                        f_i = feats[domain_i]
                        f_j = feats[domain_j]
                        loss_penalty += self.mmd(f_i, f_j)

                loss_ce /= self.ndomain
                loss_penalty /= self.ndomain * (self.ndomain - 1) / 2

                loss = loss_ce + self.tradeOff * loss_penalty

                # backward pass
                loss.backward()
                optimizer.step()
            # accumulate the loss over mini-batches.
            running_loss += loss.data

        return running_loss.item() / len(dataLoader)

    def mixup(self, trainData, optimizer, sampler=None, **kwargs):

        self.net.train()
        running_loss = 0
        dataLoader = DataLoader(trainData, batch_size=self.batchSize, sampler=sampler, drop_last=True)

        for d in dataLoader:
            with torch.enable_grad():
                # zero the parameter gradients
                optimizer.zero_grad()

                data = d[0].to(self.device).permute(0, 3, 1, 2)
                labels = d[1].type(torch.LongTensor).to(self.device)

                minibatches = [(x.to(self.device), y.to(self.device)) for x, y in
                               zip(data.chunk(self.ndomain, dim=0), labels.chunk(self.ndomain, dim=0))]

                loss = 0
                for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
                    lam = np.random.beta(self.tradeOff, self.tradeOff)
                    # lam = np.random.uniform(self.tradeOff, 1.0)
                    x = lam * xi + (1 - lam) * xj
                    predictions, _, _ = self.net.update(x)

                    loss += lam * F.cross_entropy(predictions, yi)
                    loss += (1 - lam) * F.cross_entropy(predictions, yj)

                loss /= len(minibatches)

                # backward pass
                loss.backward()
                optimizer.step()
            # accumulate the loss over mini-batches.
            running_loss += loss.data

        return running_loss.item() / len(dataLoader)

    def irm(self, trainData, optimizer, sampler=None, **kwargs):

        self.net.train()
        running_loss = 0
        dataLoader = DataLoader(trainData, batch_size=self.batchSize, sampler=sampler, drop_last=True)

        for d in dataLoader:
            with torch.enable_grad():
                # zero the parameter gradients
                optimizer.zero_grad()

                data = d[0].to(self.device).permute(0, 3, 1, 2)
                labels = d[1].type(torch.LongTensor).to(self.device)

                y_all, feats, _ = self.net.update(data)
                loss_ce = F.cross_entropy(y_all, labels)
                loss_penalty = 0
                for y_per_domain, labels_per_domain in zip(y_all.chunk(self.ndomain, dim=0),
                                                           labels.chunk(self.ndomain, dim=0)):
                    # normalize loss by domain num
                    loss_penalty += self.irm(y_per_domain, labels_per_domain) / self.ndomain

                loss = loss_ce + loss_penalty * self.tradeOff
                # backward pass
                loss.backward()
                optimizer.step()
            # accumulate the loss over mini-batches.
            running_loss += loss.data

        return running_loss.item() / len(dataLoader)

    def mldg(self, trainData, optimizer, sampler=None, **kwargs):
        r'''meta-learning based domain generalization (MLDG)
        '''
        self.net.train()
        running_loss = 0
        dataLoader = DataLoader(trainData, batch_size=self.batchSize, sampler=sampler, drop_last=True)
        n_support_domains = 3
        n_query_domains = self.ndomain - n_support_domains

        for d in dataLoader:
            with torch.enable_grad():
                # zero the parameter gradients
                # optimizer.zero_grad()
                x = d[0].to(self.device).permute(0, 3, 1, 2)
                labels = d[1].type(torch.LongTensor).to(self.device)

                x_list = x.chunk(self.ndomain, dim=0)
                labels_list = labels.chunk(self.ndomain, dim=0)
                support_domain_list, query_domain_list = random_split(x_list, labels_list, self.ndomain,
                                                                      n_support_domains)
                optimizer.zero_grad()
                with higher.innerloop_ctx(self.net, optimizer, copy_initial_weights=False) as (
                        inner_model, inner_optimizer):
                    # perform inner optimization
                    for _ in range(2):
                        loss_inner = 0
                        for (x_s, labels_s) in support_domain_list:
                            y_s, _, _ = inner_model.update(x_s)
                            # normalize loss by support domain num
                            loss_inner += F.cross_entropy(y_s, labels_s) / n_support_domains

                        inner_optimizer.step(loss_inner)

                    # calculate outer loss
                    loss_outer = 0
                    cls_acc = 0

                    # loss on support domains
                    for (x_s, labels_s) in support_domain_list:
                        y_s, _, _ = self.net.update(x_s)
                        # normalize loss by support domain num
                        loss_outer += F.cross_entropy(y_s, labels_s) / n_support_domains

                    # loss on query domains
                    for (x_q, labels_q) in query_domain_list:
                        y_q, _, _ = inner_model.update(x_q)
                        # normalize loss by query domain num
                        loss_outer += F.cross_entropy(y_q, labels_q) * self.tradeOff / n_query_domains

                loss = loss_outer
                # backward pass
                loss_outer.backward()
                optimizer.step()

            # accumulate the loss over mini-batches.
            running_loss += loss.data

        return running_loss.item() / len(dataLoader)

    def predict(self, data, sampler=None):

        predicted = []
        actual = []
        loss = 0
        self.net.eval()

        dataLoader = DataLoader(data, batch_size=self.batchSize, sampler=sampler, drop_last=False)

        # with no gradient tracking
        with torch.no_grad():
            # iterate over all the data
            for d in dataLoader:
                inputs = d[0].permute(0, 3, 1, 2).to(self.device)
                labels = d[1].type(torch.LongTensor).to(self.device)
                preds, _, _ = self.net.predict(d[0].permute(0, 3, 1, 2).to(self.device))
                n_pred = preds.shape[0]
                n_labels = labels.shape[0]
                labels = labels if n_pred == n_labels else labels.repeat_interleave(n_pred // n_labels,dim=0)
                loss += F.cross_entropy(preds, labels).item()

                # Convert the output of soft-max to class label
                _, preds_ = torch.max(preds, 1)
                predicted.extend(preds_.cpu().tolist())
                actual.extend(labels.cpu().tolist())

        return predicted, actual, loss / len(dataLoader)

    def online(self, data):

        predicted = []
        actual = []
        loss = 0
        self.net.eval()

        dataLoader = DataLoader(data, batch_size=32, sampler=None, drop_last=False)

        # with no gradient tracking
        with torch.no_grad():
            # iterate over all the data
            for d in dataLoader:
                preds, _, _ = self.net.predict(d[0].permute(0, 3, 1, 2).to(self.device))
                loss += F.cross_entropy(preds, d[1].type(torch.LongTensor).to(self.device)).data

                # Convert the output of soft-max to class label
                _, preds = torch.max(preds, 1)
                predicted.extend(preds.data.tolist())
                actual.extend(d[1].tolist())
        r = self.calculateResults(predicted, actual, classes=self.classes)
        return r

    def calculateResults(self, yPredicted, yActual, classes=None):

        acc = accuracy_score(yActual, yPredicted)
        acc = np.round(acc, 4)
        if classes is not None:
            cm = confusion_matrix(yActual, yPredicted, labels=classes)
        else:
            cm = confusion_matrix(yActual, yPredicted)

        return {'acc': acc, 'cm': cm}

    def plotLoss(self, trainLoss, valLoss, savePath=None):

        plt.figure()
        plt.title("Training Loss vs. Number of Training Epochs")
        plt.xlabel("Training Epochs")
        plt.ylabel("Loss")
        plt.plot(range(1, len(trainLoss) + 1), trainLoss, label="Train loss")
        plt.plot(range(1, len(valLoss) + 1), valLoss, label="Validation Loss")
        plt.legend(loc='upper right')
        if savePath is not None:
            plt.savefig(savePath)
        else:
            print('')
        plt.close()

    def plotAcc(self, trainAcc, valAcc, testAcc=None, savePath=None):

        plt.figure()
        plt.title("Accuracy vs. Number of Training Epochs")
        plt.xlabel("Training Epochs")
        plt.ylabel("Accuracy")
        plt.plot(range(1, len(trainAcc) + 1), trainAcc, label="Train Acc")
        plt.plot(range(1, len(valAcc) + 1), valAcc, label="Validation Acc")
        if testAcc is not None:
            plt.plot(range(1, len(testAcc) + 1), testAcc, label="test Acc")
        plt.ylim((0, 1.))
        plt.legend(loc='lower left')
        if savePath is not None:
            plt.savefig(savePath)
        else:
            print('')
        plt.close()

    def setRandom(self, seed):

        self.seed = seed

        # Set np
        np.random.seed(self.seed)

        # Set torch
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # Set cudnn
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setDevice(self, nGPU=0):

        if self.device is None:
            if self.preferedDevice == 'gpu':
                self.device = torch.device("cuda:" + str(nGPU) if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device('cpu')
            print("Code will be running on device ", self.device)

    def _findOptimizer(self, optimString):

        out = None
        if optimString in optim.__dict__.keys():
            out = optim.__dict__[optimString]
        else:
            raise AssertionError(
                'No optimizer with name :' + optimString + ' can be found in torch.optim. The list of available options in this module are as follows: ' + str(
                    optim.__dict__.keys()))
        return out

    def _findSampler(self, givenString):

        out = None
        if givenString in builtInSampler.__dict__.keys():
            out = builtInSampler.__dict__[givenString]
        elif givenString in samplers.__dict__.keys():
            out = samplers.__dict__[givenString]
        else:
            raise AssertionError('No sampler with name :' + givenString + ' can be found')
        return out

    def _findLossFn(self, lossString):

        out = None
        if lossString in nn.__dict__.keys():
            out = nn.__dict__[lossString]
        else:
            raise AssertionError(
                'No loss function with name :' + lossString + ' can be found in torch.nn. The list of available options in this module are as follows: ' + str(
                    nn.__dict__.keys()))

        return out
