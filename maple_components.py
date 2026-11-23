import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-5

class SegmentAttentionPool(nn.Module):

    def __init__(self, hidden):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))

    def forward(self, z, trial_id):
        flat = trial_id.reshape(-1)
        uniq, inv = torch.unique(flat, sorted=True, return_inverse=True)
        n_trial = uniq.numel()
        b, t, d = z.shape
        zf = z.reshape(b * t, d)
        s = self.score(zf)
        cnt = torch.zeros(n_trial, 1, device=z.device, dtype=s.dtype)
        cnt = cnt.index_add(0, inv, torch.ones_like(s))
        ssum = torch.zeros(n_trial, 1, device=z.device, dtype=s.dtype)
        ssum = ssum.index_add(0, inv, s)
        smean = ssum / cnt.clamp_min(1.0)
        ex = torch.exp((s - smean[inv]).clamp(-20, 20))
        den = torch.zeros(n_trial, 1, device=z.device, dtype=s.dtype)
        den = den.index_add(0, inv, ex)
        a = ex / den[inv].clamp_min(1e-09)
        pooled = torch.zeros(n_trial, d, device=z.device, dtype=z.dtype)
        pooled = pooled.index_add(0, inv, zf * a)
        return (pooled, inv)

class SegmentContext(nn.Module):

    def __init__(self, hidden, bottleneck=16, gamma_init=0.1):
        super().__init__()
        self.pool = SegmentAttentionPool(hidden)
        self.proj = nn.Sequential(nn.Linear(hidden, bottleneck), nn.GELU(), nn.Linear(bottleneck, hidden))
        self.gamma = nn.Parameter(torch.full((1,), float(gamma_init)))

    def forward(self, z, trial_id):
        pooled, inv = self.pool(z, trial_id)
        ctx = self.proj(pooled)[inv].view_as(z)
        return z + torch.tanh(self.gamma) * ctx

    @torch.no_grad()
    def gate_value(self):
        return float(torch.tanh(self.gamma))

def build_class_prototypes(features, labels, domain_ids, n_domain, n_class):
    d = features.shape[1]
    class_p = features.new_zeros(n_domain, n_class, d)
    counts = features.new_zeros(n_domain, n_class)
    flat = domain_ids.reshape(-1) * n_class + labels.reshape(-1)
    class_p.reshape(-1, d).index_add_(0, flat, features)
    counts.reshape(-1).index_add_(0, flat, torch.ones_like(labels, dtype=features.dtype).reshape(-1))
    return (class_p, counts)

def domain_prototypes(class_p):
    return class_p.mean(dim=1)

def subject_query(features, group_ids):
    g = group_ids.reshape(-1)
    out = torch.empty_like(features)
    for k in torch.unique(g):
        m = g == k
        out[m] = F.normalize(features[m], dim=-1, eps=1e-08).mean(dim=0, keepdim=True)
    return F.normalize(out, dim=-1, eps=1e-08)

def affinity_weights(query, domain_p, temperature=None):
    n, d = (query.shape[0], domain_p.shape[0])
    if temperature is None or not np.isfinite(temperature):
        return query.new_full((n, d), 1.0 / d)
    q = F.normalize(query, dim=-1, eps=1e-08)
    p = F.normalize(domain_p, dim=-1, eps=1e-08)
    return F.softmax(q @ p.t() / float(temperature), dim=1)

def predict_by_prototype(features, class_p, weights=None, origin=None):
    n = features.shape[0]
    if weights is None:
        proto = class_p.mean(dim=0).unsqueeze(0).expand(n, -1, -1)
    else:
        proto = torch.einsum('nd,dcf->ncf', weights, class_p)
    if origin is not None:
        features = features - origin
        proto = proto - origin.unsqueeze(1)
    fn = F.normalize(features, dim=-1, eps=1e-08)
    pn = F.normalize(proto, dim=-1, eps=1e-08)
    return F.softmax(torch.einsum('ncf,nf->nc', pn, fn), dim=1)

def masked_bce(pred, target, mask):
    loss = F.binary_cross_entropy(pred, target, reduction='none').mean(dim=-1)
    mask = mask.reshape(-1).float()
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)

def pair_target(labels_onehot):
    z = F.normalize(labels_onehot, dim=1, eps=1e-08)
    return z @ z.t()

def pairwise_bce(sim, target, mask, domain_ids=None, cross_w=1.0):
    loss = -(torch.log(sim + EPS) * target) - (1.0 - target) * torch.log(1.0 - sim + EPS)
    mask = mask.reshape(-1).float()
    if domain_ids is not None and cross_w != 1.0:
        d = domain_ids.reshape(-1)
        same = (d[:, None] == d[None, :]).float()
        w = 1.0 - target + target * (same + cross_w * (1.0 - same))
        loss = (loss * w).sum(dim=-1) / w.sum(dim=-1).clamp_min(1e-06)
    else:
        loss = loss.mean(dim=-1)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)

def cosine_matrix(x):
    z = F.normalize(x, dim=-1, eps=1e-08)
    return z @ z.t()

def aux_weight(epoch, epochs, lam0=0.5):
    return float(lam0) * max(0.0, 1.0 - float(epoch) / max(1, epochs))
