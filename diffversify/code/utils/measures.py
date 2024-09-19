from itertools import *
import tabulate
import numpy as np
from scipy.stats import mode
from sklearn.metrics import classification_report, f1_score
import torch
import logging

log = logging.getLogger(__name__)
def overlab_measure(pat,gt):
    nom = len(set(pat).intersection(set(gt)))
    denom = len(set(pat).union(set(gt)))
    measure = nom/denom
    return measure

def inner_metric(pat,gt):
    nom = len(set(pat).intersection(set(gt)))
    denom = len(set(gt))
    measure = nom/denom
    return measure

def prec(pat,gt):
    nom = len(set(pat).intersection(set(gt)))
    denom = len(set(pat))
    measure = nom/denom
    return measure

def soft_prec(P_d, P_g):
    total = []
    for p_d in P_d:
        argmax = np.max(list(map(lambda x: overlab_measure(*x), product([p_d],P_g))))
        total.append(argmax)
    return np.mean(total)

def soft_rec(P_d,P_g):
    total = []
    if len(P_d) == 0:
        return 0.000001
    for p_g in P_g:
        argmax = np.max(list(map(lambda x: overlab_measure(*x), product(P_d,[p_g]))))
        total.append(argmax)
    return np.mean(total)


def mean_compute_scores(mining,gt):
    sp_list = []
    sr_list = []
    f1_list = []
    for label in mining.keys():
        P_d = mining[label]
        P_g = gt[label]
        if len(P_d) == 0:
            sp,sr,f1 = 0,0,0
        else:
            sp = soft_prec(P_d, P_g)
            sr = soft_rec(P_d, P_g)
            if sp+sr==0:
                f1 = 0
            else:
                f1 = (2*sp*sr)/(sp+sr)
        sp_list.append(sp)
        sr_list.append(sr)
        f1_list.append(f1)
    return {'soft_prec':np.mean(sp_list), 'soft_recall':np.mean(sr_list), 'soft_F1':np.mean(f1_list)}

def compute_scores(mining,gt):
    score_dict = {}
    for label in mining.keys():
        P_d = mining[label]
        P_g = gt[label]
        if len(P_d) == 0:
            score_dict[label] = {"SP":0, "SR":0, "F1":0}
        else:
            sp = soft_prec(P_d, P_g)
            sr = soft_rec(P_d, P_g)
            if sp+sr==0:
                f1 = 0
            else:
                f1 = (2*sp*sr)/(sp+sr)
            score_dict[label] = {"SP":sp, "SR":sr, "F1":f1}
    return score_dict

def mean_compute_metric(data, labels, dict_pat, device=torch.device("cpu")):
    data = torch.tensor(data).to(device)
    labels = torch.tensor(labels).to(device)

    ret = dict()
    # Extract patterns from dict to list
    patterns = []
    pat_label = []
    for key, value in dict_pat.items():
        patterns.extend(value)
        pat_label.extend([int(key)]*len(value))

    if len(patterns) == 0:
        return ret
    # Convert patterns from indice information to 1 in data format
    binary_pats = torch.zeros((len(patterns), data.shape[1])).to(device)
    for i in range(len(patterns)):
        binary_pats[i,patterns[i]] = 1

    # Compute embedding
    embedding = torch.matmul(data, binary_pats.T)  # intersection between pattern in data
    embedding = embedding / binary_pats.sum(1)  # normalisation of intersection
    embedding = embedding.to(torch.int32)  # binarization of result : 1=pattern support

    # filtering:
    indice_pattern_keep = torch.where(embedding.sum(0) != 0)[0]
    embedding = embedding[:, indice_pattern_keep]
    # patterns = np.array(patterns, dtype='object')
    # patterns = patterns[indice_pattern_keep].tolist()

    pat_label = torch.tensor(pat_label).to(device)
    pat_label = pat_label[indice_pattern_keep]

    # Pattern count
    ret['pat_count'] = len(pat_label)
    log.debug(f"Pattern count : {len(pat_label)}")
    if len(pat_label) == 0:
        ret['cov']=0
        ret['supp']=0
        ret['purity']=0
        ret['JD_pattern']=0
        ret['wf1_quant']=0
        return ret
    # Coverage
    cov = torch.clamp(embedding.sum(1), min=0, max=1)  # Cover by line
    ret['cov'] = (cov.sum() / len(cov)).cpu().numpy()  # Normalisation by data set size

    # Mean support
    supp = embedding.sum(0)  # Number of line each pattern support
    ret['supp'] = supp.mean(dtype=torch.float).cpu().numpy()  # Average over the whole pattern set

    # Mean purity
    purity = []
    for k in torch.unique(labels):
        embedding_subset_over_k = embedding[torch.where(labels == k)[0], :]  # Extraction of the embedding information where the classe of the line is k 
        numerator = embedding_subset_over_k[:, torch.where(pat_label == k)[0]].sum(0).cpu().numpy()  # Extraction of the embedding information where the classe of the pattern is k
        denominator =  embedding[:, torch.where(pat_label == k)[0]].sum(0).cpu().numpy()  # Extraction of the embedding information where the classe of the pattern is k
        purity.extend(np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=np.float32), where=(denominator != 0)))
    ret['purity'] = np.mean(purity)

    # Mean redondancy
    def mean_sim(binary_pats):
        row_norms = torch.norm(binary_pats.to(torch.float32), dim=1, keepdim=True)
        row_norms[torch.where(row_norms == 0)] = 1e-10 
        normalized_matrix = binary_pats.to(torch.float32) / row_norms

        # Compute cosine similarity
        similarity_matrix = torch.mm(normalized_matrix, normalized_matrix.T)
        # Compute mean of similarities excluding self-similarities
        mean_similarity = (torch.sum(similarity_matrix) - torch.trace(similarity_matrix)) / (binary_pats.size(0) * (binary_pats.size(0) - 1))
        return mean_similarity
    mss = []
    for k in torch.unique(pat_label):
        mss.append(mean_sim(binary_pats[torch.where(pat_label == k)]).cpu().item())
    ret['JD_pattern'] = np.mean(mss)


    # Classification task 1, quantitative voting
    # Find the unique values and their counts
    unique_vals, counts = np.unique(labels.cpu().numpy(), return_counts=True)
    # Find the index of the most frequent value
    max_index = np.argmax(counts)
    # Get the most frequent value
    most_frequent_label = unique_vals[max_index]

    pat_label_cal = pat_label.cpu().numpy() + 1  # +1 to label for calculus
    supp_class = embedding.cpu().numpy() * pat_label_cal  # matrix of class vote
    supp_class = supp_class.astype(np.float16)  # convert in float to use NAN
    supp_class[np.where(supp_class.sum(1) == 0)] = most_frequent_label +1
    supp_class[np.where(supp_class == 0)] = np.nan  # switch 0 which are not a vote to NAN
    modes, count = mode(supp_class, axis=1, nan_policy='omit', keepdims=False)  #  Get most present vote 
    pred_label = modes - 1
    print(np.unique(pred_label, return_counts=True))
    # report = classification_report(labels, pred_label, labels=np.unique(labels), output_dict=True, zero_division=0.0)
    # wf1_quant = report['weighted avg']['f1-score']
    # print("debug")
    # print(np.isnan(pred_label).sum())
    # print(pred_label)
    ret['wf1_quant'] = f1_score(y_true=labels.cpu().numpy(), y_pred=pred_label, labels=np.unique(labels.cpu().numpy()), average='weighted', zero_division=0.0)
    # Classification task 1, qualitative voting
    # pat_label_cal = np.array(pat_label) + 1  # +1 to label for calculus
    # supp_class = embedding * pat_label_cal  # matrix of class vote
    # supp_class = supp_class.astype(np.float16)  # convert in float to use NAN
    # supp_class[np.where(supp_class == 0)] = np.nan  # switch 0 which are not a vote to NAN
    # modes, count = mode(supp_class, axis=1, nan_policy='omit', keepdims=False)  #  Get most present vote 
    # pred_label = modes - 1
    # report = classification_report(labels, pred_label, labels=np.unique(labels), output_dict=True, zero_division=0.0)
    # wf1_quant = report['weighted avg']['f1-score']

    return ret


    

def overlap_function(mining, gt):
    scores = {}
    for k2, v2 in mining.items():
        res = {}
        res = {i:{"index": -1, "overlap":-1.0} for i in range(len(gt[k2]))}
        res["Avg Overlap"] = 0.0
        if len(v2)>0:
            acc_overlaps = 0.0
            for i, pat in enumerate(gt[k2]):
                
                overlaps = list(map(lambda x: overlab_measure(*x), product([pat],v2)))
                pos = np.argmax(overlaps)
                max_overlap = np.max(overlaps)
                if max_overlap > 0.0:
                    res[i] = {"index": pos, "overlap":max_overlap}
                    acc_overlaps += max_overlap
                else:
                    res[i] = {"index": -1, "overlap":-1.0}
            res["Avg Overlap"] = acc_overlaps/len(gt[k2])
        scores[k2] = res
    return scores

def mean_overlap_function(mining, gt):
    scores = overlap_function(mining, gt)
    eval_list = []
    for k,v in scores.items():
        eval_list.append(v["Avg Overlap"])
    return {'JD_gtVSpats':np.mean(eval_list)}




