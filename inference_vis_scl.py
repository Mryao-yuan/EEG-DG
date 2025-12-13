import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from torch.utils.data import DataLoader

from dataset.eegDataset import eegDataset
from utils.tools import get_transform
from utils.NeuronCLustering import NeuronClustering, get_norm

from network.networks import B7
from collections import OrderedDict

import mne

class ConceptInferenceB7:
    def __init__(self, model_path_scl,model_path_coco,  device='cuda:0',model_type=None):
        self.device = torch.device(device)
        self.clusters = None
        self.weights = None
        self.model_type = model_type
        self.NACT = None 

       
        model_path = model_path_coco if model_type == 'coco' else model_path_scl
        print(f"Loading B7 model from: {model_path}")
        # inputSize: [Bands, Chans, Time] -> BCI42a 通常是 [9, 22, 1000] (FilterBank=9)
        # 注意：这里的 inputSize 只是为了初始化全连接层，具体维度会被网络自动适应
        self.net = B7(inputSize=[9, 22, 1000], nClass=4, m=FEATURE_DIM, isProj=True)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('network_sma.', '')
            name = name.replace('network.', '')
            name = name.replace('module.', '')
            new_state_dict[name] = v
        self.net.load_state_dict(new_state_dict)
        self.net.to(self.device)
        self.net.eval()
        
        if self.model_type == 'coco':
            print(">> Mode is CoCo: Initializing Clustering Module for Re-clustering...")
            self.NACT = NeuronClustering(
                label_ids=range(4),
                reg_layer_list=['feature'],
                quantile=0.05,  # 推理时可以用 0.05
                act_ratio=0.05, # 推理时可以用 0.05
                num_concept_clusters=5,
                clustering_method='kmeans'
            )
            self.clusters = None # 标记为 None，后续 visualize_topk 会检测并触发计算
        elif self.model_type == 'scl':
            print(">> Mode is SCL: Creating Identity Clusters (1 Neuron = 1 Concept).")
            num_neurons = 32 # B7 的 m 参数, init m 个概念，每个概念一个神经元
            self.clusters = {
                'feature': torch.arange(num_neurons).unsqueeze(1).to(self.device)
            }
            self.weights = {
                'feature': torch.ones(num_neurons, 1).to(self.device)
            }
        
    def load_test_data(self, sub_id):
        print(f"Loading Test Data for Subject {sub_id}...")
        label_path = os.path.join(DATA_ROOT, 'dataLabels.csv')
        
        full_data = eegDataset(dataPath=DATA_ROOT, dataLabelsPath=label_path, 
                               preloadData=False, transform=None) 
        # 筛选特定被试的 idx
        # data.labels 结构: [count_id, filename, label, subject, session]
        sub_str = str(sub_id).zfill(3)
        test_idx = [i for i, x in enumerate(full_data.labels) if x[3] == sub_str]
        full_data.createPartialDataset(test_idx, loadNonLoadedData=True)
        return full_data

    def recompute_clusters(self, loader):
        print("Re-computing clusters using current data...")
        d_layer_output = []
        
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device).permute(0, 3, 1, 2)
                _, feats, _ = self.net(x) # B7 forward
                if isinstance(feats, (list, tuple)): feats = feats[-1]
                # 归一化
                feat = get_norm(feats).cpu()
                d_layer_output.append({"feature": feat, "labels": y, "accs": None})


        self.NACT.compute_neuron_cluster(d_layer_output, padding=True, unact=False)
        
        # 提取结果
        if len(self.NACT.layer_act_neuron_dict['feature'][0]) == 0:
            print("!!! Warning: Re-clustering failed (empty). Check quantile/act_ratio.")
            # 如果聚类失败，回退到 SCL 模式 (防止报错)
            num_neurons = 32
            self.clusters = {'feature': torch.arange(num_neurons).unsqueeze(1).to(self.device)}
            self.weights = {'feature': torch.ones(num_neurons, 1).to(self.device)}
        else:
            self.clusters = {l: torch.stack(v[0], dim=0).to(self.device) 
                             for l, v in self.NACT.layer_act_neuron_dict.items()}
            self.weights = {l: torch.stack(v[1], dim=0).to(self.device) 
                            for l, v in self.NACT.layer_act_neuron_dict.items()}
            print(f">>> Clusters re-computed successfully. Shape: {self.clusters['feature'].shape}")

    def visualize_topk(self, loader, k=10, save_dir='./viz_output/',sub_id=None):
        if not os.path.exists(save_dir+f'{self.model_type}/{sub_id}'): os.makedirs(save_dir+f'{self.model_type}/{sub_id}')
        
        # 确保有聚类中心
        if self.clusters is None and self.model_type == 'coco':
            self.recompute_clusters(loader)
        target_layer = 'feature'
        cluster_indices = self.clusters[target_layer]
        cluster_weights = self.weights[target_layer]
        num_concepts = cluster_indices.shape[0]
        # print(f"Number of concepts: {num_concepts}")
        topk_storage = [[] for _ in range(num_concepts)] # 20 个聚类的概念
        
        print("Scanning samples for Top-K...")
        with torch.no_grad():
            for x, y in loader:
                x_gpu = x.to(self.device).permute(0, 3, 1, 2)
                _, feats, _ = self.net(x_gpu)
                if isinstance(feats, (list, tuple)): z = feats[-1]
                else: z = feats
                z = get_norm(z)
                
                feature_dim = z.shape[1]
                
                # 计算激活分
                for c_id in range(num_concepts):
                    idx = cluster_indices[c_id]
                    w = cluster_weights[c_id]
                    valid_mask = (idx < feature_dim) & (idx >= 0) & (w > 0)
                    if valid_mask.sum() == 0: continue
                    
                    valid_idx = idx[valid_mask].long()
                    valid_w = w[valid_mask]
                    # 获取概念激活分数
                    scores = (z[:, valid_idx] * valid_w.unsqueeze(0)).sum(dim=1)     
                    # 存入
                    x_cpu = x.numpy()

                    for i in range(len(scores)):
                        topk_storage[c_id].append({
                            'score': scores[i].item(),
                            'data': x_cpu[i], 
                            'label': y[i].item()
                        })      
        self._plot_bands(topk_storage, k, save_dir,sub_id)
        
        
    def _plot_bands(self, topk_storage, k, save_dir,sub_id):
        """
        绘制多频带脑地形图 (Theta, Alpha, Beta)，输入数据[Batch, 22, 1000, 9] 数据格式,9:频带
        """
        ch_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 
                    'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
        label_names = ['Left', 'Right', 'Foot', 'Tongue']
        montage = mne.channels.make_standard_montage('standard_1020')
        info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types='eeg')
        info.set_montage(montage)

        for c_id, c_list in enumerate(topk_storage):
            c_list.sort(key=lambda x: x['score'], reverse=True) # 降序
            samples = c_list[:k] # 获取 Top-K 样本,即每个概念簇下最强激活的 K 个样本
            if len(samples) == 0: continue
            
            # 1. 数据聚合
            # 原始数据 x 的形状是 [22, 1000, 9] (Chans, Time, Bands)
            # stack 后: [K, 22, 1000, 9]
            stack_data = np.stack([s['data'] for s in samples])
            
            # --- 2. 提取 Alpha 频带的时域波形 ---
            # Alpha 是 Band 1 (索引 1)
            # 取出 Alpha 频带数据 -> [K, 22, 1000]
            alpha_data_k = stack_data[:, :, :, 1] 
            # 对样本求平均 -> [22, 1000]
            alpha_wave = np.mean(alpha_data_k, axis=0) 
            
            # --- 3. 计算各频带能量 ---
            # 先对 Time (dim 2) 求方差(能量) -> [K, 22, 9]
            power_data = np.var(stack_data, axis=2)
            # 再对 Samples (dim 0) 求平均 -> [22, 9] (每个通道、每个频带的平均能量)
            avg_power = np.mean(power_data, axis=0)

            # 提取特定频带能量 (最后一维是 Bands)
            theta_power = avg_power[:, 0]        # Band 0: 4-8Hz
            alpha_power = avg_power[:, 1]        # Band 1: 8-12Hz
            # Band 2-6: 12-32Hz (Beta), 对频带维求平均
            beta_power = np.mean(avg_power[:, 2:7], axis=1) 

            # 统计标签
            labels = [s['label'] for s in samples]
            counts = np.bincount(labels, minlength=4)
            dom_idx = np.argmax(counts)
            dom_name = label_names[dom_idx]
            purity = counts[dom_idx] / len(samples)
            # === 绘图 ===
            fig, axes = plt.subplots(1, 4, figsize=(20, 4))
            fig.suptitle(f"Concept #{c_id}: Frequency Analysis (Dominant: {dom_name} {purity:.0%})", fontsize=16)
            
            # Subplot 1: Alpha Waveform (C3/C4)
            times = np.linspace(0, 4, 1000)
            # C3=7, C4=11 用于 验证对侧效应
            axes[0].plot(times, alpha_wave[7], 'b', label='C3 (Left Hemi)') # 左脑电极，负责右手
            axes[0].plot(times, alpha_wave[11], 'r', label='C4 (Right Hemi)') # 右脑电极，负责左手
            axes[0].set_title("Avg Alpha Waveform (8-12Hz)")
            axes[0].legend(loc='lower right')
            axes[0].set_xlabel('Time (s)')

            # Subplots 2-4: Topomaps
            # Theta
            im, _ = mne.viz.plot_topomap(theta_power, info, axes=axes[1], show=False, cmap='RdBu_r')
            axes[1].set_title("Theta (4-8 Hz)")
            plt.colorbar(im, ax=axes[1])
            # Alpha
            im, _ = mne.viz.plot_topomap(alpha_power, info, axes=axes[2], show=False, cmap='RdBu_r')
            axes[2].set_title("Alpha/Mu (8-12 Hz)")
            plt.colorbar(im, ax=axes[2])
            # Beta
            im, _ = mne.viz.plot_topomap(beta_power, info, axes=axes[3], show=False, cmap='RdBu_r')
            axes[3].set_title("Beta (12-30 Hz)")
            plt.colorbar(im, ax=axes[3])

            plt.tight_layout()
            save_path = os.path.join(save_dir +f'{self.model_type}/{sub_id}',f'freq_viz_c{c_id}_{dom_name}.png')
            plt.savefig(save_path)
            plt.close()
            print(f"Saved {save_path}")
    

if __name__ == '__main__':

    TARGET_SUB = 2  
    DATA_ROOT = '/home/yaoyuan/EEG/Datasets/bci42a/multiviewPython' 
    model_path_coco = f'output/bci42a/2025-12-08--11-27-B7_coco_sr_0.2_weight_0.1/sub{TARGET_SUB}\\checkpoint.pth.tar'
    model_path_scl = f'./output/bci42a/2025-12-05--12-11-B7_SCL/sub{TARGET_SUB}\\checkpoint.pth.tar'
    DATASET_ID = 0  # bci42a
    FEATURE_DIM = 32
    DEVICE = 'cuda:0'
    
    inference = ConceptInferenceB7( model_path_scl,model_path_coco, model_type='coco', device=DEVICE) 
    
    test_data = inference.load_test_data(sub_id=TARGET_SUB)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    inference.visualize_topk(test_loader, k=10,sub_id=TARGET_SUB)