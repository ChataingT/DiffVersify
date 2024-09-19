import sys
sys.path.append("../")
import numpy as np
import pandas as pd
from itertools import *
import torch

from method.diffnaps import *
import numpy as np
from utils.measures import *
import method.my_layers as myla
import method.my_loss as mylo
from method.diffnaps import *

import logging

log = logging.getLogger(__name__)

import json


from method.network import *


import matplotlib.pyplot as plt
from scipy.special import entr
from sklearn.ensemble import IsolationForest


def tuple_to_name(string):
    a,b,c,_ = string.split(",")
    string = a[1:] + "_"+b[2]+"_"+c[0]
    return string


def gen_label_dict(base, length):
    label_list = list(map(lambda x: "".join(map(str,x)),product(base, repeat=length)))
    str_to_label = dict(zip(label_list,range(0,len(label_list))))
    return str_to_label, {v:k for k,v in str_to_label.items()}

def tranform_labels(labels, label_dict):
    num_labels = []
    for row in labels:
        l = "".join(list(map(str, list(row))))
        num_labels.append(label_dict[l])
    return num_labels


def sort_str(string):
    pattern = list(map(int,string.split(" ")))
    for p in sorted(pattern):
        print(p, end=" ")
        
def to_list(string):
    pattern = list(map(int,string.split(" ")))
    return pattern


def get_positional_patterns(weights, classifier_weights, t1=0.3, t2=0.3, t_mean=0.25, general=False, device=torch.device("cpu")):
    l = []
    num_l = []
    all_patterns = []
    # extract all patterns present in the data using thresholding on encoder
    hidden = torch.zeros(weights.shape[0], dtype=torch.int32).to(device)
    for i,hn in enumerate(myla.BinarizeTensorThresh(weights, t1)):
        pat = torch.squeeze(hn.nonzero())
        pat = pat.reshape(-1)
        if hn.sum() >= 1 and list(pat.cpu().numpy()) not in l and weights[i].cpu().numpy().mean()<t_mean:
            all_patterns.append(list(pat.cpu().numpy()))
            l.append((i,list(pat.cpu().numpy())))
            num_l.append((i,list(weights[i].cpu().numpy())))
            hidden[i] = 1
    all_patterns = set(map(tuple, all_patterns)) 

    # assign patterns using thresholding on the classifer              
    patterns = dict(l)
    num_patterns = dict(num_l)
    bin_class = myla.BinarizeTensorThresh(classifier_weights, t2).to(device)
    assignment_tensor = (hidden * bin_class)
    labels = [str(i) for i in range(classifier_weights.shape[0])]
    assignment = {k:{} for k in labels }
    num_assignment = {k:{} for k in labels }
    assigned_patterns = []
    for key,hn in zip(labels, assignment_tensor):
        temp = sorted(list(map(list,set( [tuple(patterns.get(int(i), -1)) for i in torch.squeeze(hn.nonzero()).reshape(-1)]))))
        sorted(temp,key=lambda x:x[0])
        assignment[key] = temp
        num_assignment[key] = sorted(list(map(list,set( [tuple(num_patterns.get(int(i), -1)) for i in torch.squeeze(hn.nonzero()).reshape(-1)]))))
        assigned_patterns.extend(temp)
    assigned_patterns = set(map(tuple,assigned_patterns))
    general_patterns = list(map(list,all_patterns-assigned_patterns))
    if general:
        return l,num_l, hidden, num_assignment, assignment, sorted(general_patterns)
    else:
        return l,num_l, hidden, num_assignment, assignment
    


def get_positional_patterns_binaps(weights, data, labels, general=False):
    patterns = {str(k):[] for k in np.unique(labels)}

    # extract all patterns present in the data using thresholding on encoder
    w = torch.round_(weights + 0.3)
    embedding = ((np.matmul(data, w.T) / w.sum(1)).nan_to_num(0) >= 1).to(torch.int16)

    for i, hn in enumerate(w):
        pat = torch.squeeze(hn.nonzero()).numpy()
        if hn.sum() >= 2:
            max_c = 0
            cc = 0
            for c in np.unique(labels):
                supps = embedding[np.array(np.where(labels == c)), i].sum()
                if supps > max_c:
                    cc = c
                    max_c = supps
            patterns[str(cc)].append(pat)
    if general:
        return None,None, None, None, patterns, None
    else:
        return None,None, None, None, patterns
    


def draw_positional_patterns(weights, classifier_weights, nbr_of_draw=1,  t1=0.3, t2=0.3, t_mean=0.25, general=False):
    l = []
    num_l = []
    all_patterns = []
    # extract all patterns present in the data using thresholding on encoder
    hidden = torch.zeros(weights.shape[0], dtype=torch.int32) 
    # for i,hn in enumerate(myla.BinarizeTensorThresh(weights, t1)):
    for i,hn in enumerate(weights):
        for times in range(nbr_of_draw):
            pat = np.random.binomial(1, hn)
            pat = torch.squeeze(hn.nonzero())
            pat = pat.reshape(-1)
            if hn.sum() >= 1 and list(pat.cpu().numpy()) not in l and weights[i].cpu().numpy().mean()<t_mean:
                all_patterns.append(list(pat.cpu().numpy()))
                l.append((i,list(pat.cpu().numpy())))
                num_l.append((i,list(weights[i].cpu().numpy())))
                hidden[i] = 1
    all_patterns = set(map(tuple, all_patterns)) 

    # assign patterns using thresholding on the classifer              
    patterns = dict(l)
    num_patterns = dict(num_l)
    bin_class = myla.BinarizeTensorThresh(classifier_weights, t2)
    assignment_tensor = (hidden * bin_class)
    labels = [str(i) for i in range(classifier_weights.shape[0])]
    assignment = {k:{} for k in labels }
    num_assignment = {k:{} for k in labels }
    assigned_patterns = []
    for key,hn in zip(labels, assignment_tensor):
        temp = sorted(list(map(list,set( [tuple(patterns.get(int(i), -1)) for i in torch.squeeze(hn.nonzero()).reshape(-1)]))))
        sorted(temp,key=lambda x:x[0])
        assignment[key] = temp
        num_assignment[key] = sorted(list(map(list,set( [tuple(num_patterns.get(int(i), -1)) for i in torch.squeeze(hn.nonzero()).reshape(-1)]))))
        assigned_patterns.extend(temp)
    assigned_patterns = set(map(tuple,assigned_patterns))
    general_patterns = list(map(list,all_patterns-assigned_patterns))
    if general:
        return l,num_l, hidden, num_assignment, assignment, sorted(general_patterns)
    else:
        return l,num_l, hidden, num_assignment, assignment
    


class TrainConfig():
    def __init__(self, train_set_size = 0.9, batch_size = 64, test_batch_size = 64, epochs = 100, lr = 0.01,
                       gamma = 0.1, seed = 1, log_interval = 10, hidden_dim = 500, thread_num = 12,
                       weight_decay = 0.05, wd_class=0.00, binaps=False, lambda_c=1.0,
                       spike_weight=0.0, vertical_decay=0.0, sparse_regu=0.0, elb_lamb=0.0, elb_k=0.0,
                       class_elb_k = 0.0, class_elb_lamb = 0.0, regu_rate = 1.0, class_regu_rate = 1.0,model=DiffnapsNet, 
                       loss=mylo.weightedXor, alpha=0,t1=0.1, t2=0.1,
                        test=False, init_enc="", save_xp=False, k_f=15, k_w=15):
        self.train_set_size = train_set_size
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.gamma = gamma
        self.alpha = alpha
        self.seed = seed
        self.log_interval = log_interval
        self.hidden_dim = hidden_dim
        self.thread_num = thread_num
        self.weight_decay = weight_decay
        self.wd_class = wd_class
        self.binaps = binaps
        self.lambda_c = lambda_c
        self.model = model
        self.loss = loss
        self.spike_weight = spike_weight
        self.vertical_decay = vertical_decay
        self.sparse_regu = sparse_regu
        self.aggregator = torch.mean
        self.test = test
        self.elb_k = elb_k
        self.elb_lamb = elb_lamb
        self.class_elb_k = class_elb_k
        self.class_elb_lamb = class_elb_lamb
        self.regu_rate = regu_rate
        self.class_regu_rate = class_regu_rate
        self.init_enc = init_enc
        self.save_xp = save_xp
        self.k_f = k_f
        self.t1=t1
        self.t2=t2
        self.k_w = k_w

    def lazy_load_from_dict(self,dico: dict):
        for key, value in dico.items():
            if key in self.__dict__.keys():
                setattr(self, key, value)
            else :
                log.debug(f"{key} not in TrainConfig")
                
        if dico['method'] == 'binaps':
            self.binaps = True
        # elif dico['method'] == 'diffnaps':
        elif dico['method'] == 'diffversify':
            self.loss = mylo.weightedXorCover
        
        return self
                
                

    def to_str(self):
        ret = self.__dict__.copy()
        ret['aggregator'] = 'mean'
        ret['loss'] = ret['loss'].__name__
        ret['model'] = ret['model'].__name__

        return json.dumps(ret)


def check_frequencies(data, patterns_gt, slack=0):
    ret = {}
    for k, patterns in patterns_gt.items():
        l = []
        for pattern in patterns:
            support = np.sum(np.sum(data[:,pattern],axis=1)>=(len(pattern)-slack))
            l.append(support)
        ret[k] = l
    return ret

def leave_x_out_frequencies(data, patterns_gt, x):
    ret = {}
    for k, patterns in patterns_gt.items():
        l = []
        for pattern in patterns:
            support = np.sum(np.sum(data[:,pattern],axis=1)==(len(pattern)-x))
            l.append(support)
        ret[k] = l
    return ret


def assign_pat(mp, labels, data):
    patterns = {str(k.item()):[] for k in torch.unique(labels)}
    embedding = ((torch.matmul(data, mp.T) / mp.sum(1)).nan_to_num(0) >= 1).to(torch.int16)
    pc = []

    for hn in mp:
        pat = list(torch.squeeze(hn.nonzero(), dim=1).cpu().numpy())
        pc.append(pat)

    class_vote = torch.zeros((mp.shape[0], len(patterns.keys())))
    for c in torch.unique(labels):
        class_vote[:, c.item()] = embedding[torch.where(labels == c)[0], :].sum(0).reshape(-1)
    classes = torch.argmax(class_vote, dim=1)

    for c, pat in zip(classes.cpu().numpy(), pc):
        if len(pat) >= 1:
            patterns[str(c)].append(pat)
    
    return patterns


def compile_new_pat_whole(labels, patterns, data, device, n=2, max_iter=100, rank=15):
    log.info('Compile wew pattern with the whole strategy')
    labels = torch.tensor(labels).to(device)
    data = torch.tensor(data).to(device)

    if type(n) != list:
        n=[n]
    new_p = {k:dict() for k in n}

    pc = []
    for c in torch.unique(labels):
        pc.extend(patterns[str(c.item())])

    if len(pc) <= 1:
        return {k:patterns for k in n}
    mp = torch.zeros((len(pc), data.shape[1]), dtype=torch.float, device=device)        
    
    for i, id in enumerate(pc):
        mp[i, id] = 1

    em = (torch.matmul(data, mp.T) / mp.sum(1) >= 1).to(torch.float)
    W, H = nmf_binary(em, rank=rank, gpu=device, max_iter=max_iter)

    
    for k in n:
        _, indices = torch.topk(H, k, dim=1, largest=True)
        H_top = torch.zeros((H.shape[0], k), dtype=int, device=device)
        for i in range(H.shape[0]):
            H_top[i, :] = indices[i, :]

        nmp = torch.zeros((mp.shape[0]+len(H_top), mp.shape[1])).to(device)
        nmp[:mp.shape[0], :] = mp
        for i, it in enumerate(H_top):
            nmp[i+mp.shape[0], :] = mp[it, :].sum(0)
        nmp = nmp.clamp_max(1)

        new_p[k] = assign_pat(mp=nmp, labels=labels, data=data)

    return new_p

def compile_new_pat_by_class(labels, patterns, data, device, n=2, max_iter=100, rank=15):
    log.info('Compile new pattern with the filter strategy')

    labels = torch.tensor(labels).to(device)
    data = torch.tensor(data).to(device)

    if type(n) != list:
        n = [n]

    new_p = {k:dict() for k in n}

    for c in torch.unique(labels):
        pc = patterns[str(c.item())]
        if len(pc) <=1:
            for k in n:
                new_p[k][str(c.item())] = pc
            continue

        sub_data = data[torch.where(labels == c.item())]

        mp = torch.zeros((len(pc), sub_data.shape[1]), dtype=torch.float, device=device)        
        
        for i, id in enumerate(pc):
            mp[i, id] = 1

        em = (torch.matmul(sub_data, mp.T) / mp.sum(1) >= 1).to(torch.float)
        W, H = nmf_binary(em, rank=rank, gpu=device, max_iter=max_iter)



        for k in n:
            if k > len(pc):
                new_p[k][str(c.item())] = pc
                continue
            
            new_p[k][str(c.item())] = []
            _, indices = torch.topk(H, k, dim=1, largest=True)

            H_top = torch.zeros((H.shape[0], k), dtype=int, device=device)
            for i in range(H.shape[0]):
                H_top[i, :] = indices[i, :]

            nmp = torch.zeros((mp.shape[0]+len(H_top), mp.shape[1])).to(device)
            nmp[:mp.shape[0], :] = mp

            for i, it in enumerate(H_top):
                nmp[i+mp.shape[0], :] = mp[it, :].sum(0)
            nmp=nmp.clamp_max(1)

            for hn in nmp:
                pat = list(torch.squeeze(hn.nonzero(), dim=1).cpu().numpy())
                if len(pat) >= 1:
                    new_p[k][str(c.item())].append(pat)

    return new_p

def nmf_binary(V, rank, max_iter=100, tol=1e-4, gpu=torch.device("cpu")):
    """
    Non-negative Matrix Factorization for binary data using multiplicative updates.

    Parameters:
        V (torch.Tensor): Input binary data matrix of shape (n_samples, n_features).
        rank (int): Number of components.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.

    Returns:
        W (torch.Tensor): Basis matrix of shape (n_samples, rank).
        H (torch.Tensor): Encoding matrix of shape (rank, n_features).
    """
    V.to(gpu)
    n_samples, n_features = V.shape

    # Initialize random matrices
    W = torch.rand(n_samples, rank, device=gpu)
    H = torch.rand(rank, n_features, device=gpu)

    # Normalize the initial matrices
    W /= W.sum(dim=1, keepdim=True)
    H /= H.sum(dim=0, keepdim=True)

    for i in range(max_iter):
        # Update H
        WH = torch.mm(W, H)
        WH[WH == 0] = 1e-10  # Avoid division by zero
        H *= torch.mm(W.t(), (V / WH)) / torch.mm(W.t(), torch.ones_like(V))

        # Update W
        WH = torch.mm(W, H)
        WH[WH == 0] = 1e-10  # Avoid division by zero
        W *= torch.mm((V / WH), H.t()) / torch.mm(torch.ones_like(V), H.t())

        # Compute Frobenius norm of the residual
        residual = torch.norm(V - torch.mm(W, H))
        if residual < tol:
            log.debug('NMF stoped by threshold')
            break
    log.debug(f"NMF for {rank} Residual : {residual}")
    return W, H
