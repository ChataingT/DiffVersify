import torch
import torch.nn as nn
import math

class VanillaWeightedXor(nn.Module):

    def __init__(self, weight, weight_decay, device_gpu):
        super(VanillaWeightedXor, self).__init__()
        ## sparsity of data
        self.weight = weight
        ## decay rate
        self.weight_decay = weight_decay
        print("Data Sparsity:")
        print(self.weight)

    def forward(self, output, target, w):

        relu = nn.ReLU()
        diff = relu((output - target)).sum(1).mul(self.weight).mean() + relu((target - output)).sum(1).mul(1-self.weight).mean()
        diff += self.weight_decay*(((w - 1/target.size()[1])).sum(1).clamp(min=1).pow(2).sum())

        return diff



## weigh the different cases (positive vs negative) differently
## based on the data sparsity
class weightedXor(nn.Module):

    def __init__(self, weight, weight_decay, device_gpu, aggregator=torch.sum, label_decay=0.05, labels=0, **kwargs):
        super(weightedXor, self).__init__()
        ## sparsity of data
        self.weight = weight
        ## decay rate
        self.weight_decay = weight_decay
        self.labels = labels
        self.split_dim = -1
        print("Data Sparsity:")
        print(self.weight)

    def forward(self, output, target, w, hidden=None, **kwargs):
        relu = nn.ReLU()
        xor = relu((output - target)).sum(1).mul(self.weight).mean() + relu((target - output)).sum(1).mul(1-self.weight).mean()

        horizon_l2 = horizontal_L2(self.config.weight_decay, w, target)

        vl2 = 0
        if self.config.vertical_decay>0:
            vl2 = vertical_L2(self.config.vertical_decay, w, target)

        elb_reg = 0
        if self.config.elb_k >0.0 or self.config.elb_lamb>0.0:
            elb_reg = elb_regu(self.config.elb_k, self.config.elb_lamb, w, target)

        sparse_reg = 0
        if not hidden is None:
            sparse_reg = torch.mean(torch.abs(hidden))

        details = {'xor': xor,
        'horizon_l2': horizon_l2,
        'vertical_l2': vl2,
        'elb_regu': elb_reg,
        f'sparse_regu*{self.config.sparse_regu}': sparse_reg,
        }

        diff = xor + horizon_l2 + vl2 + elb_reg + self.config.sparse_regu * sparse_reg

        return diff, details
  
## weigh the different cases (positive vs negative) differently
## based on the data sparsity
class weightedXorVanilla(nn.Module):

    def __init__(self, weight, weight_decay, device_gpu, aggregator=torch.sum, label_decay=0.05, labels=0, **kwargs):
        super(weightedXorVanilla, self).__init__()
        ## sparsity of data
        self.weight = weight
        ## decay rate
        self.weight_decay = weight_decay
        self.labels = labels
        self.split_dim = -1
        print("Data Sparsity:")
        print(self.weight)

    def forward(self, output, target, w, hidden=None, **kwargs):
        relu = nn.ReLU()
        xor = relu((output - target)).sum(1).mul(self.weight).mean() + relu((target - output)).sum(1).mul(1-self.weight).mean()


        details = {'xor': xor,
        }

        diff = xor

        return diff, details
  

## weigh the different cases (positive vs negative) differently
## based on the data sparsity
class weightedXorCover(nn.Module):

    def __init__(self, weight, weight_decay, device_gpu, aggregator=torch.sum, label_decay=0.05, labels=0, alpha=0, **kwargs):
        super(weightedXorCover, self).__init__()
        ## sparsity of data
        self.weight = weight
        ## decay rate
        self.weight_decay = weight_decay
        self.labels = labels
        self.split_dim = -1
        self.alpha = alpha
        print("Data Sparsity:")
        print(self.weight)

    def forward(self, output, target, w, hidden=None, **kwargs):
        relu = nn.ReLU()
        xor = relu((output - target)).sum(1).mul(self.weight).mean() + relu((target - output)).sum(1).mul(1-self.weight).mean()

        horizon_l2 = horizontal_L2(self.config.weight_decay, w, target)

        vl2 = 0
        if self.config.vertical_decay>0:
            vl2 = vertical_L2(self.config.vertical_decay, w, target)

        elb_reg = 0
        if self.config.elb_k >0.0 or self.config.elb_lamb>0.0:
            elb_reg = elb_regu(self.config.elb_k, self.config.elb_lamb, w, target)

        sparse_reg = 0
        if not hidden is None:
            sparse_reg = torch.mean(torch.abs(hidden))

        # Covering with orthogonality of patterns
        cd = 0
        if self.alpha > 0:
            cd = contrastive_divergence_loss(w)

        diff = xor + horizon_l2 + vl2 + elb_reg + self.config.sparse_regu * sparse_reg + self.alpha * cd

        details = {'xor': xor,
        'horizon_l2': horizon_l2,
        'vertical_l2': vl2,
        'elb_regu': elb_reg,
        f'sparse_regu*{self.config.sparse_regu}': sparse_reg,
        f'ortogonal*{self.alpha}': cd
        }

        return diff, details

class weightedXorUni(nn.Module):

    def __init__(self, weight, weight_decay, device_gpu, aggregator=torch.sum, label_decay=0.05, labels=0, alpha=0, **kwargs):
        super(weightedXorUni, self).__init__()
        ## sparsity of data
        self.weight = weight
        ## decay rate
        self.weight_decay = weight_decay
        self.labels = labels
        self.split_dim = -1
        self.alpha = alpha
        print("Data Sparsity:")
        print(self.weight)

    def forward(self, output, target, w, hidden=None):
        relu = nn.ReLU()


        xor = relu((output - target)).sum(1).mul(self.weight).mean() + relu((target - output)).sum(1).mul(1-self.weight).mean()

        horizon_l2 = horizontal_L2(self.config.weight_decay, w, target)

        vl2 = 0
        if self.config.vertical_decay>0:
            vl2 = vertical_L2(self.config.vertical_decay, w, target)

        elb_reg = 0
        if self.config.elb_k >0.0 or self.config.elb_lamb>0.0:
            elb_reg = elb_regu(self.config.elb_k, self.config.elb_lamb, w, target)

        sparse_reg = 0
        if not hidden is None:
            sparse_reg = torch.mean(torch.abs(hidden))

        # Covering with orthogonality of patterns
        cd = 0
        if self.alpha > 0:
            cd = cosine_similarity_matrix(w).sum(0) / w.shape[0]
            cd = (cd * hidden).sum(1).mean()

        diff = xor + horizon_l2 + vl2 + elb_reg + self.config.sparse_regu * sparse_reg + self.alpha * cd

        details = {'xor': xor,
        'horizon_l2': horizon_l2,
        'vertical_l2': vl2,
        'elb_regu': elb_reg,
        f'sparse_regu*{self.config.sparse_regu}': sparse_reg,
        f'ortogonal*{self.alpha}': cd
        }

        return diff, details

# @torch.compile
def cosine_similarity(t1, t2, dim=-1, eps=1e-8):
    # get normalization value
    t1_div = torch.linalg.vector_norm(t1, dim=dim, keepdims=True)
    t2_div = torch.linalg.vector_norm(t2, dim=dim, keepdims=True)

    t1_div = t1_div.clone()
    t2_div = t2_div.clone()
    with torch.no_grad():
        t1_div.clamp_(math.sqrt(eps))
        t2_div.clamp_(math.sqrt(eps))

    # normalize, avoiding division by 0
    t1_norm = t1 / t1_div
    t2_norm = t2 / t2_div

    return (t1_norm * t2_norm).sum(dim=dim)   

def cosine_similarity_matrix(matrix):
    # Normalize rows (vectors)
    row_norms = torch.norm(matrix, dim=1, keepdim=True)
    normalized_matrix = matrix / row_norms

    # Compute cosine similarity
    similarity_matrix = torch.mm(normalized_matrix, normalized_matrix.T)
    return similarity_matrix

def contrastive_divergence_loss(matrix):
    # matrix shape : nbr_pattern * nbr_feature
    # Compute pairwise cosine similarity
    # similarity_matrix = torch.nn.functional.cosine_similarity(matrix.unsqueeze(1), matrix.unsqueeze(0), dim=-1)  # rajouter une dim
    similarity_matrix = cosine_similarity_matrix(matrix)  # rajouter une dim

    # Compute mean of similarities excluding self-similarities
    mean_similarity = (torch.sum(similarity_matrix) - torch.trace(similarity_matrix)) / (matrix.size(0) * (matrix.size(0) - 1))

    # Penalize similarity
    loss = mean_similarity

    return loss
    
def vertical_L2(lambd, w, target):
    return torch.mean(lambd*(((w - 1/target.size()[1])).sum(1).clamp(min=1).pow(2)))

def spike_regu(lambd, w, target):
    return lambd * torch.mean(w*(1-w))

def horizontal_L2(lambd,w, target):
    #print(((w - 1/target.size()[1])).sum(1).clamp(min=1))
    #print(((w - 1/target.size()[1])).sum(1))
    return torch.mean(lambd*(((w - 1/target.size()[1])).sum(1).clamp(min=1).pow(2)))

def horizontal_L2_class(lambd,w, target):
    return torch.mean(lambd*(((w - 0)).sum(1).clamp(min=1).pow(2)))

def horizontal_L1(lambd, w, target):
    return torch.mean(torch.abs(lambd*(((w - 1/target.size()[1])).sum(1).clamp(min=1))))

def elb_regu(k, lambd, w, target):
    offset = 1/target.shape[1]
    w = w - offset
    elastic = lambda w: k*torch.abs(w) + lambd * torch.square(w)
    #elastic = lambda w: horizontal_L1(k,w,target) + horizontal_L2(lambd,w,target)# lambd * torch.square(w)
    return torch.mean( torch.minimum(elastic(w), elastic(w-1)))

def elb_regu_class(k, lambd, w, target):
    elastic = lambda w: k*torch.abs(w) + lambd * torch.square(w)
    return torch.mean( torch.minimum(elastic(w), elastic(w-1)))

class xor(nn.Module):

    def __init__(self, weight_decay, device_gpu):
        super(xor, self).__init__()
        self.weight_decay = weight_decay

    def forward(self, output, target, w):
        diff = (output - target).pow(2).sum(1).mean()

        # set minimum of weight to 0, to avoid penalizing too harshly for large matrices
        diff += (w - 1/target.size()[1]).pow(2).sum()*self.weight_decay

        return diff
