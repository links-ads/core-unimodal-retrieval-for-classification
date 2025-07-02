import torch
import numpy as np
import faiss
    
def dot_score(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))


def cos_sim(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

def l2(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the L2 distance between vectors in a and b.
    For search compatibility, returns negative distances so that higher values are better.
    :return: Matrix with res[i][j] = -L2_distance(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    
    # Reshape if needed
    if len(a.shape) == 1: 
        a = a.unsqueeze(0)
    if len(b.shape) == 1: 
        b = b.unsqueeze(0)
    
    # Compute squared L2 distances
    a_norm = (a**2).sum(1).view(-1, 1)
    b_norm = (b**2).sum(1).view(1, -1)
    dist = a_norm + b_norm - 2.0 * torch.mm(a, b.transpose(0, 1))
    
    # Return negative distances so higher values are better (for compatibility with retrieval)
    return -dist