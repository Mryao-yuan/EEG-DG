import torch
from sklearn.metrics.pairwise import euclidean_distances
import torch.nn as nn


class MMDLoss(nn.Module):

    def __init__(self):
        super(MMDLoss, self).__init__()

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        Kxx = self.gaussian_kernel(f_s, f_s).mean()
        Kyy = self.gaussian_kernel(f_t, f_t).mean()
        Kxy = self.gaussian_kernel(f_s, f_t).mean()
        return Kxx + Kyy - 2 * Kxy


class MMD_loss(nn.Module):
    r'''
    EEG-DG：A Multi-Source Domain Generalization Framework for Motor Imagery EEG Classification
    https://ieeexplore.ieee.org/document/10609514
    '''
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, eps=1e-8):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type
        self.eps = eps

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples + self.eps)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / (bandwidth_temp + self.eps)) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.shape[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss
        
class Dist_Loss(nn.Module):
    r'''
    EEG-DG：A Multi-Source Domain Generalization Framework for Motor Imagery EEG Classification
    https://ieeexplore.ieee.org/document/10609514
    '''
    def __init__(self, classes=4):
        super(Dist_Loss, self).__init__()
        self.classes = classes


    def intraclass_compactness(self, data, labels):
        unique_labels = torch.unique(labels)
        compactness = 0

        for label in unique_labels:
            class_data = data[labels == label]

            class_data = class_data.cpu().detach()

            class_data_np = class_data.numpy()

            distances = euclidean_distances(class_data_np)
            compactness += torch.sum(torch.from_numpy(distances)) / 2

        return compactness / data.shape[0]


    def interclass_separability(self, data, labels):
        unique_labels = torch.unique(labels)
        separability = 0

        for i in range(len(unique_labels)):
            for j in range(len(unique_labels)):
                if i != j:
                    class_data_1 = data[labels == unique_labels[i]]
                    class_data_2 = data[labels == unique_labels[j]]

                    class_data_1 = class_data_1.cpu().detach()
                    class_data_2 = class_data_2.cpu().detach()

                    class_data_1_np = class_data_1.numpy()
                    class_data_2_np = class_data_2.numpy()

                    distances = euclidean_distances(class_data_1_np, class_data_2_np)
                    separability += torch.sum(torch.from_numpy(distances))

        return separability / data.shape[0]

    def compute_class_centers(self, data, labels):
        unique_labels = torch.unique(labels)
        class_centers = []

        for label in unique_labels:
            class_data = data[labels == label]
            class_center = torch.mean(class_data, dim=0)
            class_centers.append(class_center)

        return class_centers


    def forward(self, source_data, source_labels, target_data, target_labels, alpha):
        dist_1 = self.intraclass_compactness(source_data, source_labels) - alpha * self.interclass_separability(
            source_data, source_labels)
        dist_2 = self.intraclass_compactness(target_data, target_labels) - alpha * self.interclass_separability(
            target_data, target_labels)

        source_centers = self.compute_class_centers(source_data, source_labels)
        target_centers = self.compute_class_centers(target_data, target_labels)

        source_centers_np = torch.stack(source_centers).detach().cpu().numpy()
        target_centers_np = torch.stack(target_centers).detach().cpu().numpy()

        matrix = torch.cdist(torch.from_numpy(source_centers_np), torch.from_numpy(target_centers_np), p=2)

        diagonal = torch.diag(matrix)
        dist_31 = torch.mean(diagonal)

        dist = dist_1 + dist_2 + dist_31

        return dist
