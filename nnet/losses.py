#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
from itertools import permutations


def loss_calc(est, ref, loss_type):
    """
    time-domain loss: sisdr
    """
    # time domain (wav input)
    if loss_type == "sisdr":
        loss = batch_SDR_torch(est, ref)
    if loss_type == "mse":
        loss = batch_mse_torch(est, ref)
    if loss_type == "log_mse":
        loss = batch_log_mse_torch(est, ref)


    return loss


def calc_sdr_torch(estimation, origin, mask=None):
    """
    batch-wise SDR caculation for one audio file on pytorch Variables.
    estimation: (batch, nsample)
    origin: (batch, nsample)
    mask: optional, (batch, nsample), binary
    """
    
    if mask is not None:
        origin = origin * mask
        estimation = estimation * mask
    
    origin_power = torch.pow(origin, 2).sum(1, keepdim=True) + 1e-8  # (batch, 1)
    
    scale = torch.sum(origin*estimation, 1, keepdim=True) / origin_power  # (batch, 1)
    
    est_true = scale * origin  # (batch, nsample)
    est_res = estimation - est_true  # (batch, nsample)
    
    true_power = torch.pow(est_true, 2).sum(1)
    res_power = torch.pow(est_res, 2).sum(1)
    
    return 10*torch.log10(true_power) - 10*torch.log10(res_power)  # (batch, 1)


def batch_SDR_torch(estimation, origin, mask=None):
    """
    batch-wise SDR caculation for multiple audio files.
    estimation: (batch, nsource, nsample)
    origin: (batch, nsource, nsample)
    mask: optional, (batch, nsample), binary
    """
    
    batch_size_est, nsource_est, nsample_est = estimation.size()
    batch_size_ori, nsource_ori, nsample_ori = origin.size()
    
    assert batch_size_est == batch_size_ori, "Estimation and original sources should have same shape."
    assert nsource_est == nsource_ori, "Estimation and original sources should have same shape."
    assert nsample_est == nsample_ori, "Estimation and original sources should have same shape."
    
    assert nsource_est < nsample_est, "Axis 1 should be the number of sources, and axis 2 should be the signal."
    
    batch_size = batch_size_est
    nsource = nsource_est
    nsample = nsample_est
    
    # zero mean signals
    estimation = estimation - torch.mean(estimation, 2, keepdim=True).expand_as(estimation)
    origin = origin - torch.mean(origin, 2, keepdim=True).expand_as(estimation)
    
    # possible permutations
    perm = list(set(permutations(np.arange(nsource))))
    
    # pair-wise SDR
    SDR = torch.zeros((batch_size, nsource, nsource)).type(estimation.type())
    for i in range(nsource):
        for j in range(nsource):
            SDR[:,i,j] = calc_sdr_torch(estimation[:,i], origin[:,j], mask)
    
    # choose the best permutation
    SDR_max = []
    SDR_perm = []
    for permute in perm:
        sdr = []
        for idx in range(len(permute)):
            sdr.append(SDR[:,idx,permute[idx]].view(batch_size,-1))
        sdr = torch.sum(torch.cat(sdr, 1), 1)
        SDR_perm.append(sdr.view(batch_size, 1))
    SDR_perm = torch.cat(SDR_perm, 1)
    SDR_max, _ = torch.max(SDR_perm, dim=1)
    
    return - SDR_max / nsource

# def calc_mse_torch(estimation, origin):
#     return torch.mean(torch.pow(estimation-origin,2),1).mean(1)

def batch_mse_torch(estimation, origin):
    """
    batch-wise mse caculation for multiple audio files.
    estimation: (batch, nsource, frames, freq_bins)
    origin: (batch, nsource, frames, freq_bins)
    nsource = 2
    """
    mse1 = torch.sqrt(torch.pow(estimation - origin, 2).mean([3])).mean([1,2])
    mse2 = torch.sqrt(torch.pow(estimation - origin.flip([1]), 2).mean([3])).mean([1,2])
    return torch.stack((mse1, mse2),1).min(1)[0]

def batch_log_mse_torch(estimation, origin):
    """
    batch-wise mse caculation for multiple audio files.
    estimation: (batch, nsource, frames, freq_bins)
    origin: (batch, nsource, frames, freq_bins)
    nsource = 2
    """
    # eps = 1e-20
    # mse1 = torch.log10(torch.sqrt(torch.pow(estimation - origin, 2).mean([3])).mean([1,2])+eps)
    # mse2 = torch.log10(torch.sqrt(torch.pow(estimation - origin.flip([1]), 2).mean([3])).mean([1,2])+eps)
    mse1 = torch.log10(torch.pow(estimation - origin, 2).mean([3])).mean([1,2])
    mse2 = torch.log10(torch.pow(estimation - origin.flip([1]), 2).mean([3])).mean([1,2])
    return torch.stack((mse1, mse2),1).min(1)[0]

if __name__ == "__main__":
    est = torch.rand(10, 2, 32, 1000)
    ref = torch.rand(10, 2, 32, 1000)

    out = loss_calc(est, ref, "mse")
    print(out.shape)
    print(out)
    