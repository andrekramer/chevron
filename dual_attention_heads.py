# v16_final_combined.py — Dual-Path Counterfactual Transformer (final + plots)
# -----------------------------------------------------------------------------
# What you get in one file:
# - Baseline and DualPath (x-attn to counterfactuals)
# - OOD rehearsal with fused supervision (train what we evaluate)
# - Split calibration: ID = VectorScaling, OOD = classwise Temp-only
# - Split entropy floors (ID/OOD)
# - Mixer β cap on OOD + gentle gate nudging
# - Full evaluation (aggregate + ID/OOD) and per-path diagnostics
# - Metrics JSON + CSV
# - Plots:
#     • Reliability (ID/OOD) with bin-count bars
#     • Confidence & Entropy histograms (ID vs OOD)
#     • Gate g, β, JS histograms (ID vs OOD)
#     • OOD scatter: β vs JS (colored by correctness)
#     • Risk–Coverage and Accuracy–Confidence curves
#     • Confusion matrix
#     • Compact 2×3 “panel_summary.png” + “panel_confusion.png”
# -----------------------------------------------------------------------------

import os, math, random, argparse, json, csv
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ------------------------- Config & Globals ----------------------------------

seed = 1337
random.seed(seed); torch.manual_seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class CFG:
    # data sizes
    train_size:int=6000; val_size:int=1000; test_size:int=2000
    min_len:int=3; max_len:int=6; batch_size:int=128
    # model
    d_model:int=128; nhead:int=4; nlayers:int=2; dim_ff:int=256; dropout:float=0.10
    # opt
    lr:float=3e-4; epochs:int=12; weight_decay:float=0.01
    label_smoothing:float=0.05
    # dual path training
    gate_commit_lambda:float=0.02; repr_div_lambda:float=0.03
    aux_loss_lambda_f:float=0.15; aux_loss_lambda_c:float=0.45
    gate_temperature:float=0.6; factual_droppath_p:float=0.15
    cf_aug_p:float=0.20; cf_warmup_epochs:int=5
    # mixer
    tau_conf:float=0.7; disagree_mix_lambda:float=0.34
    expert_temp:float=2.2; fused_conf_pivot:float=1.4; fused_conf_slope:float=3.0
    # OOD rehearsal
    ood_batch_size:int=48
    # split entropy floors
    id_target_entropy:float=1.10; id_max_eps:float=0.007
    ood_target_entropy:float=1.40; ood_max_eps:float=0.022
    # vector-scaling L2
    vs_l2:float=5e-4

CFG = CFG()

# ------------------------- Toy Opposites World -------------------------------

OPPOSITE_PAIRS = [
    ('hot','cold'),('up','down'),('light','dark'),('open','closed'),
    ('left','right'),('sun','moon'),('wet','dry'),('near','far'),
    ('full','empty'),('high','low'),
]
OPERATORS = ['ID','NOT','SWAP']
words = sorted({w for a,b in OPPOSITE_PAIRS for w in (a,b)})
itos = words + OPERATORS + ['<PAD>']
stoi = {t:i for i,t in enumerate(itos)}
PAD_IDX = stoi['<PAD>']

def opposite_token(tok:str)->str:
    d = dict(OPPOSITE_PAIRS + [(b,a) for a,b in OPPOSITE_PAIRS])
    return d.get(tok, tok)

def make_counterfactual(seq):
    return [t if (t in OPERATORS or t=='<PAD>') else opposite_token(t) for t in seq]

def apply_rule(ctx, op):
    if op=='ID': return ctx[-1]
    if op=='NOT': return opposite_token(ctx[-1])
    if op=='SWAP':
        if len(ctx)>=2: return (ctx[:-2]+[ctx[-1],ctx[-2]])[-1]
        return ctx[-1]
    return ctx[-1]

ALL_LASTS = [w for a,b in OPPOSITE_PAIRS for w in (a,b)]
random.shuffle(ALL_LASTS)
HELD_OUT=set()
for op in ['NOT','SWAP']:
    held=set(ALL_LASTS[:4])    # unseen (op,last) combos at train → surprise at test
    HELD_OUT|={(op,t) for t in held}
    random.shuffle(ALL_LASTS)

def sample_sequence():
    L=random.randint(CFG.min_len, CFG.max_len)
    content=[random.choice(words) for _ in range(L-1)]
    op=random.choice(OPERATORS)
    y=apply_rule(content, op)
    return content+[op], y, op

def seq_to_ids(seq): return [stoi[t] for t in seq]

class OppDataset(Dataset):
    def __init__(self, size, split):
        self.samples=[]
        while len(self.samples)<size:
            seq,y,op=sample_sequence()
            last_tok=seq[-2]
            is_held=(op,last_tok) in HELD_OUT
            if split=='train' and is_held: continue
            self.samples.append((seq,y))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        seq,y=self.samples[idx]
        cf_seq=make_counterfactual(seq)
        return seq_to_ids(seq), seq_to_ids(cf_seq), stoi[y]

def collate(batch):
    seqs, cfseqs, ys = zip(*batch)
    maxlen=max(len(s) for s in seqs)
    def pad(seq): return seq + [PAD_IDX]*(maxlen-len(seq))
    X = torch.tensor([pad(s) for s in seqs])
    CFX = torch.tensor([pad(s) for s in cfseqs])
    Y = torch.tensor(ys)
    return X, CFX, Y

train_dl = DataLoader(OppDataset(CFG.train_size,'train'), batch_size=CFG.batch_size, shuffle=True,  collate_fn=collate)
val_dl   = DataLoader(OppDataset(CFG.val_size,'val'),     batch_size=CFG.batch_size, shuffle=False, collate_fn=collate)
test_dl  = DataLoader(OppDataset(CFG.test_size,'test'),   batch_size=CFG.batch_size, shuffle=False, collate_fn=collate)
vocab_size=len(itos)

# ------------------------- Model ---------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        pe=torch.zeros(max_len, d_model)
        pos=torch.arange(0,max_len, dtype=torch.float).unsqueeze(1)
        div=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return self.dropout(x + self.pe[:, :x.size(1), :])

class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        D=CFG.d_model; H=CFG.nhead; FF=CFG.dim_ff; p=CFG.dropout
        self.emb=nn.Embedding(vocab_size, D, padding_idx=PAD_IDX)
        self.pos=PositionalEncoding(D, p)
        enc=nn.TransformerEncoderLayer(D,H,FF,p,batch_first=True)
        self.enc=nn.TransformerEncoder(enc, CFG.nlayers)
        self.ln=nn.LayerNorm(D); self.fc=nn.Linear(D, vocab_size)
    def forward(self, x):
        mask=(x==PAD_IDX)
        h=self.enc(self.pos(self.emb(x)), src_key_padding_mask=mask)
        z=self.ln(h[:,-1,:])
        return self.fc(z)

class DualPathXAttn(nn.Module):
    def __init__(self):
        super().__init__()
        D=CFG.d_model; H=CFG.nhead; FF=CFG.dim_ff; p=CFG.dropout
        self.tau = CFG.gate_temperature
        self.emb_f = nn.Embedding(vocab_size, D, padding_idx=PAD_IDX)
        self.emb_c = nn.Embedding(vocab_size, D, padding_idx=PAD_IDX)
        self.pos = PositionalEncoding(D, p)
        encF = nn.TransformerEncoderLayer(D, H, FF, p, batch_first=True)
        encC = nn.TransformerEncoderLayer(D, H, int(1.25*FF), p, batch_first=True)
        self.f_enc = nn.TransformerEncoder(encF, CFG.nlayers)
        self.c_enc = nn.TransformerEncoder(encC, CFG.nlayers + 1)
        self.x_fq_c = nn.MultiheadAttention(D, H, dropout=p, batch_first=True)
        self.x_cq_f = nn.MultiheadAttention(D, H, dropout=p, batch_first=True)
        self.post_f = nn.Sequential(nn.Linear(D, FF), nn.GELU(), nn.Dropout(p), nn.Linear(FF, D))
        self.post_c = nn.Sequential(nn.Linear(D, int(1.25*FF)), nn.GELU(), nn.Dropout(p), nn.Linear(int(1.25*FF), D))
        self.ln_f = nn.LayerNorm(D); self.ln_c = nn.LayerNorm(D)
        self.gate  = nn.Sequential(nn.Linear(2*D, D), nn.GELU(), nn.Linear(D, 1))
        self.f_proj= nn.Linear(D, D, bias=False)
        self.c_proj= nn.Linear(D, D, bias=False)
        self.out_ln= nn.LayerNorm(D)
        self.fc    = nn.Linear(D, vocab_size)
        self.fc_f  = nn.Linear(D, vocab_size)
        self.fc_c  = nn.Linear(D, vocab_size)
    def forward(self, x, xcf):
        padf=(x==PAD_IDX); padc=(xcf==PAD_IDX)
        hf = self.f_enc(self.pos(self.emb_f(x)),  src_key_padding_mask=padf)
        hc = self.c_enc(self.pos(self.emb_c(xcf)), src_key_padding_mask=padc)
        hf_last = hf[:, -1:, :]; hc_last = hc[:, -1:, :]
        af,_ = self.x_fq_c(query=hf_last, key=hc, value=hc, key_padding_mask=padc)
        ac,_ = self.x_cq_f(query=hc_last, key=hf, value=hf, key_padding_mask=padf)
        hf_enh = self.ln_f(hf_last + af + self.post_f(af))
        hc_enh = self.ln_c(hc_last + ac + self.post_c(ac))
        hf_s = hf_enh.squeeze(1); hc_s = hc_enh.squeeze(1)
        g = torch.sigmoid(self.gate(torch.cat([hf_s, hc_s], -1)) / self.tau).squeeze(-1)
        if self.training and CFG.factual_droppath_p > 0:
            keep = (torch.rand(hf_s.size(0), device=hf_s.device) > CFG.factual_droppath_p).float().unsqueeze(-1)
            hf_s = hf_s * keep
        fused = self.f_proj(hf_s)*(1-g.unsqueeze(-1)) + self.c_proj(hc_s)*g.unsqueeze(-1)
        fused = self.out_ln(fused)
        logits   = self.fc(fused)
        logits_f = self.fc_f(hf_s)
        logits_c = self.fc_c(hc_s)
        return logits, g, logits_f, logits_c, hf_s, hc_s

# ------------------------- Utils ---------------------------------------------

def ece_score(probs, labels, n_bins=15):
    conf, pred = probs.max(-1)
    acc = pred.eq(labels)
    ece = torch.zeros(1, device=probs.device)
    bins=torch.linspace(0,1,n_bins+1,device=probs.device)
    for i in range(n_bins):
        m=(conf>bins[i]) & (conf<=bins[i+1])
        if m.sum()>0:
            ece += m.float().mean() * (conf[m].mean() - acc[m].float().mean()).abs()
    return ece.item()

def ids_to_tokens(seq_ids): return [itos[i] for i in seq_ids if i!=PAD_IDX]

def batch_cf_targets(xb):
    t=[]
    for ids in xb.detach().cpu().tolist():
        toks=ids_to_tokens(ids)
        if len(toks)<2: t.append(stoi[toks[-1]] if toks else PAD_IDX); continue
        content,op=toks[:-1],toks[-1]
        cf=[opposite_token(t) if t not in OPERATORS else t for t in content]
        t.append(stoi[apply_rule(cf, op)])
    return torch.tensor(t, dtype=torch.long, device=xb.device)

def surprise_mask(xb):
    m=[]
    for ids in xb.cpu().tolist():
        seq=[itos[i] for i in ids if i!=PAD_IDX]
        if len(seq)<2: m.append(False); continue
        last,op=seq[-2],seq[-1]; m.append((op,last) in HELD_OUT)
    return torch.tensor(m, dtype=torch.bool)

def symmetric_js(p,q,eps=1e-9):
    p=p.clamp_min(eps); q=q.clamp_min(eps); m=0.5*(p+q)
    return 0.5*((p*(p/m).log()).sum(-1) + (q*(q/m).log()).sum(-1))

class CESmooth(nn.Module):
    def __init__(self, smoothing=0.0): super().__init__(); self.s=smoothing
    def forward(self, logits, target, num_classes):
        logp=F.log_softmax(logits, -1)
        with torch.no_grad():
            td=torch.full_like(logp, self.s/(num_classes-1))
            td.scatter_(1, target.unsqueeze(1), 1.0-self.s)
        return -(td*logp).sum(-1).mean()
ce_smooth=CESmooth(CFG.label_smoothing)

def cosine_similarity(a,b,eps=1e-8):
    an=a/(a.norm(dim=-1,keepdim=True)+eps); bn=b/(b.norm(dim=-1,keepdim=True)+eps)
    return (an*bn).sum(-1)

def maybe_cf_augment(x, xcf, y, cf_aug_p):
    if random.random()<cf_aug_p:
        y_cf=batch_cf_targets(x)
        return torch.cat([x,xcf],0), torch.cat([xcf,xcf],0), torch.cat([y,y_cf],0)
    return x, xcf, y

def synth_ood_batch(batch_size=32):
    xs=[]; xcs=[]; ys=[]
    for _ in range(batch_size):
        seq,y,op = sample_sequence()
        last_tok = seq[-2]
        op = random.choice(['NOT','SWAP'])
        if (op,last_tok) not in HELD_OUT:
            pair = next(iter(HELD_OUT))[1]
            seq[-2] = pair
        seq[-1] = op
        y = apply_rule(seq[:-1], op)
        xs.append(seq_to_ids(seq)); xcs.append(seq_to_ids(make_counterfactual(seq))); ys.append(stoi[y])
    maxlen=max(len(s) for s in xs)
    def pad_to(ms): return [s + [PAD_IDX]*(maxlen-len(s)) for s in ms]
    return (torch.tensor(pad_to(xs), device=device),
            torch.tensor(pad_to(xcs), device=device),
            torch.tensor(ys, device=device))

def freeze_factual(model, freeze=True):
    for m in [model.emb_f, model.f_enc, model.post_f, model.fc_f, model.f_proj]:
        for p in m.parameters(): p.requires_grad = not freeze

# ------------------------- Mixer (differentiable, OOD-aware) ------------------

def anchored_mix(logits, logits_f, logits_c, sm_mask=None):
    fused = F.softmax(logits, dim=-1)
    p_f = F.softmax(logits_f / CFG.expert_temp, dim=-1)
    p_c = F.softmax(logits_c / CFG.expert_temp, dim=-1)

    eps=1e-9
    Hf = -(p_f * (p_f.clamp_min(eps)).log()).sum(-1)
    Hc = -(p_c * (p_c.clamp_min(eps)).log()).sum(-1)
    conf = torch.stack([-Hf, -Hc], dim=-1) / CFG.tau_conf
    a = F.softmax(conf, dim=-1)[:, 1]  # CF share inside expert mix

    m = 0.5*(p_f+p_c)
    js = 0.5*((p_f*(p_f/(m+eps)).clamp_min(eps).log()).sum(-1) + (p_c*(p_c/(m+eps)).clamp_min(eps).log()).sum(-1))
    beta_js = torch.sigmoid(3.0 * (js - js.mean()))

    Hfused = -(fused * (fused.clamp_min(eps)).log()).sum(-1)
    conf_brake = torch.sigmoid(CFG.fused_conf_slope * (Hfused - CFG.fused_conf_pivot))
    beta = torch.clamp(CFG.disagree_mix_lambda * beta_js * conf_brake, 0.0, 1.0)

    # OOD cap & CF damp
    if sm_mask is not None:
        sm = sm_mask.to(beta.device)
        if sm.any():
            beta = beta.clone()
            beta[sm] = torch.minimum(beta[sm], torch.full_like(beta[sm], 0.15))
            a = a.clone()
            a[sm] = 0.5 * a[sm]

    expert_mix = (1 - a.unsqueeze(-1)) * p_f + a.unsqueeze(-1) * p_c
    probs = (1 - beta.unsqueeze(-1)) * fused + beta.unsqueeze(-1) * expert_mix
    return probs

# ------------------------- Train / Eval / Calibrate ---------------------------

def train(model, dual_flag=False):
    opt=torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    if dual_flag: freeze_factual(model, True)
    for ep in range(1, CFG.epochs+1):
        model.train(); total=0.0
        cf_aug = max(CFG.cf_aug_p, 0.6) if (dual_flag and ep<=CFG.cf_warmup_epochs) else CFG.cf_aug_p
        aux_lambda_f = (0.05 if (dual_flag and ep<=CFG.cf_warmup_epochs) else CFG.aux_loss_lambda_f)
        aux_lambda_c = (CFG.aux_loss_lambda_c if (dual_flag and ep>CFG.cf_warmup_epochs) else 0.80)
        if dual_flag and ep==CFG.cf_warmup_epochs+1: freeze_factual(model, False)

        for xb, xcfb, yb in train_dl:
            xb, xcfb, yb = xb.to(device), xcfb.to(device), yb.to(device)
            if dual_flag: xb, xcfb, yb = maybe_cf_augment(xb, xcfb, yb, cf_aug)
            opt.zero_grad()

            if not dual_flag:
                logits = model(xb)
                loss = ce_smooth(logits, yb, vocab_size)
            else:
                logits, g, lf, lc, hf, hc = model(xb, xcfb)
                loss_main = ce_smooth(logits, yb, vocab_size)
                commit = (0.25 - (g-0.5).pow(2)).mean()
                div = cosine_similarity(hf, hc).mean()
                y_cf = batch_cf_targets(xb)
                aux = aux_lambda_f*ce_smooth(lf, yb, vocab_size) + aux_lambda_c*ce_smooth(lc, y_cf, vocab_size)
                loss = loss_main + CFG.gate_commit_lambda*commit + CFG.repr_div_lambda*div + aux

                if ep > CFG.cf_warmup_epochs:
                    xb_ood, xcfb_ood, yb_ood = synth_ood_batch(batch_size=CFG.ood_batch_size)
                    sm_ood = surprise_mask(xb_ood.cpu())
                    logits_ood, g_ood, lf_ood, lc_ood, _, _ = model(xb_ood, xcfb_ood)
                    ycf_ood = batch_cf_targets(xb_ood)
                    probs_ood = anchored_mix(logits_ood, lf_ood, lc_ood, sm_mask=sm_ood)
                    loss_ood = (
                        0.70 * ce_smooth(probs_ood.log(), yb_ood,  vocab_size) +
                        0.20 * ce_smooth(lf_ood,           yb_ood,  vocab_size) +
                        0.10 * ce_smooth(lc_ood,           ycf_ood, vocab_size)
                    )
                    with torch.no_grad():
                        pf_ood = F.softmax(lf_ood / CFG.expert_temp, dim=-1)
                        pc_ood = F.softmax(lc_ood / CFG.expert_temp, dim=-1)
                        js_ood = symmetric_js(pf_ood, pc_ood)
                        high_js = (js_ood > js_ood.mean()).float()
                    gate_push = ((g_ood - 0.55).pow(2) * high_js).mean()
                    loss = loss + loss_ood + 0.01 * gate_push

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * xb.size(0)

        vm = evaluate(model, val_dl, dual_flag=dual_flag, scaler_id=None, scaler_ood=None, use_split_floor=False)
        print(f"Epoch {ep:02d} | train_loss={total/CFG.train_size:.4f} | "
              f"val_acc={vm['acc']:.3f} | val_ece={vm['ece']:.3f} | val_ent={vm['entropy']:.3f}")
    return model

class VectorScaler(nn.Module):
    def __init__(self, num_classes, init_T=1.0):
        super().__init__()
        self.logT = nn.Parameter(torch.full((num_classes,), math.log(init_T)))
        self.bias = nn.Parameter(torch.zeros(num_classes))
    def forward(self, logits):
        T = self.logT.exp().clamp(min=1.0)
        return logits / T.unsqueeze(0) + self.bias.unsqueeze(0)

class ClasswiseTempScaler(nn.Module):
    def __init__(self, num_classes, init_T=1.0):
        super().__init__()
        self.logT = nn.Parameter(torch.full((num_classes,), math.log(init_T)))
    def forward(self, logits):
        T = self.logT.exp().clamp(min=1.0)
        return logits / T.unsqueeze(0)

@torch.no_grad()
def probs_to_logits_safe(p, eps=1e-12):
    return (p.clamp_min(eps)).log()

@torch.no_grad()
def split_val_probs(model, dl):
    model.eval()
    probs_id=[]; y_id=[]; probs_ood=[]; y_ood=[]
    for xb, xcfb, yb in dl:
        xb, xcfb, yb = xb.to(device), xcfb.to(device), yb.to(device)
        logits, _, lf, lc, _, _ = model(xb, xcfb)
        sm = surprise_mask(xb.cpu())
        probs = anchored_mix(logits, lf, lc, sm_mask=sm)
        if (~sm).sum()>0:
            idx=(~sm).to(probs.device)
            probs_id.append(probs[idx].detach().cpu()); y_id.append(yb[idx].detach().cpu())
        if sm.sum()>0:
            idx=sm.to(probs.device)
            probs_ood.append(probs[idx].detach().cpu()); y_ood.append(yb[idx].detach().cpu())
    P_id = torch.cat(probs_id,0) if probs_id else None
    Y_id = torch.cat(y_id,0) if y_id else None
    P_ood= torch.cat(probs_ood,0) if probs_ood else None
    Y_ood= torch.cat(y_ood,0) if y_ood else None
    return P_id, Y_id, P_ood, Y_ood

def fit_vs_from_probs(P, Y, num_classes, l2=5e-4, max_iter=300, lr=0.05):
    if P is None or len(P)==0: return None
    L = probs_to_logits_safe(P.to(device)); Y = Y.to(device)
    scaler = VectorScaler(num_classes=num_classes, init_T=1.0).to(device)
    opt = torch.optim.LBFGS(scaler.parameters(), lr=lr, max_iter=max_iter)
    def brier_loss(logits, y):
        p = F.softmax(logits, dim=-1)
        onehot = torch.zeros_like(p).scatter_(1, y.unsqueeze(1), 1.0)
        return (p - onehot).pow(2).sum(dim=1).mean()
    def closure():
        opt.zero_grad()
        LT = scaler(L)
        loss = brier_loss(LT, Y) + l2 * (scaler.bias.pow(2).sum() + scaler.logT.pow(2).sum())
        loss.backward(); return loss
    opt.step(closure); return scaler

def fit_temp_from_probs(P, Y, num_classes, max_iter=300, lr=0.05):
    if P is None or len(P)==0: return None
    L = probs_to_logits_safe(P.to(device)); Y = Y.to(device)
    scaler = ClasswiseTempScaler(num_classes=num_classes, init_T=1.0).to(device)
    opt = torch.optim.LBFGS(scaler.parameters(), lr=lr, max_iter=max_iter)
    def brier_loss(logits, y):
        p = F.softmax(logits, dim=-1)
        onehot = torch.zeros_like(p).scatter_(1, y.unsqueeze(1), 1.0)
        return (p - onehot).pow(2).sum(dim=1).mean()
    def closure():
        opt.zero_grad(); loss = brier_loss(scaler(L), Y); loss.backward(); return loss
    opt.step(closure); return scaler

@torch.no_grad()
def apply_split_scaling(probs, xb, scaler_id, scaler_ood):
    sm = surprise_mask(xb.cpu())
    L = probs_to_logits_safe(probs).to(device)
    LT = L.clone()
    idx_id  = (~sm).to(L.device)
    idx_ood = ( sm).to(L.device)
    if scaler_id is not None and idx_id.any():
        LT[idx_id]  = scaler_id(L[idx_id])
    if scaler_ood is not None and idx_ood.any():
        LT[idx_ood] = scaler_ood(L[idx_ood])
    return F.softmax(LT, dim=-1)

@torch.no_grad()
def apply_split_entropy_floor(probs, xb,
                              id_target=CFG.id_target_entropy, id_eps=CFG.id_max_eps,
                              ood_target=CFG.ood_target_entropy, ood_eps=CFG.ood_max_eps):
    H = -(probs * (probs.clamp_min(1e-9)).log()).sum(-1)
    sm = surprise_mask(xb.cpu()).to(probs.device)
    out = probs.clone()
    V = probs.size(-1)
    if (~sm).any():
        idx=(~sm); ramp=torch.sigmoid(3.0*(id_target - H[idx])); eps=id_eps*ramp
        out[idx]=(1 - eps.unsqueeze(-1))*out[idx] + eps.unsqueeze(-1)*(1.0/V)
    if sm.any():
        idx=sm; ramp=torch.sigmoid(3.0*(ood_target - H[idx])); eps=ood_eps*ramp
        out[idx]=(1 - eps.unsqueeze(-1))*out[idx] + eps.unsqueeze(-1)*(1.0/V)
    return out

@torch.no_grad()
def evaluate(model, dl, dual_flag=False, scaler_id=None, scaler_ood=None, use_split_floor=True):
    model.eval()
    tot=0; ok=0; ents=[]; probs_all=[]; labels_all=[]
    for xb, xcfb, yb in dl:
        xb, xcfb, yb = xb.to(device), xcfb.to(device), yb.to(device)
        if not dual_flag:
            probs = F.softmax(model(xb), dim=-1)
        else:
            logits, _, lf, lc, _, _ = model(xb, xcfb)
            sm = surprise_mask(xb.cpu())
            probs = anchored_mix(logits, lf, lc, sm_mask=sm)
            if (scaler_id is not None) or (scaler_ood is not None):
                probs = apply_split_scaling(probs, xb, scaler_id, scaler_ood)
            if use_split_floor:
                probs = apply_split_entropy_floor(probs, xb)
        pred = probs.argmax(-1)
        ok += (pred==yb).sum().item(); tot += yb.size(0)
        ent = -(probs * (probs.clamp_min(1e-9)).log()).sum(-1)
        ents += ent.tolist()
        probs_all.append(probs.cpu()); labels_all.append(yb.cpu())
    probs_cat=torch.cat(probs_all,0); labels_cat=torch.cat(labels_all,0)
    return {'acc': ok/tot, 'entropy': float(torch.tensor(ents).mean()), 'ece': ece_score(probs_cat, labels_cat, 15)}

@torch.no_grad()
def eval_split(model, dl, dual_flag=False, scaler_id=None, scaler_ood=None, use_split_floor=True):
    model.eval()
    res={'ID':{'tot':0,'ok':0,'ents':[],'probs':[],'labels':[]},
         'OOD':{'tot':0,'ok':0,'ents':[],'probs':[],'labels':[]}}
    for xb, xcfb, yb in dl:
        xb, xcfb, yb = xb.to(device), xcfb.to(device), yb.to(device)
        if not dual_flag:
            probs = F.softmax(model(xb), dim=-1)
        else:
            logits, _, lf, lc, _, _ = model(xb, xcfb)
            sm = surprise_mask(xb.cpu())
            probs = anchored_mix(logits, lf, lc, sm_mask=sm)
            if (scaler_id is not None) or (scaler_ood is not None):
                probs = apply_split_scaling(probs, xb, scaler_id, scaler_ood)
            if use_split_floor:
                probs = apply_split_entropy_floor(probs, xb)
        pred = probs.argmax(-1); ent = -(probs * (probs.clamp_min(1e-9)).log()).sum(-1)
        sm = surprise_mask(xb.cpu())
        for key,mask in [('ID',~sm),('OOD',sm)]:
            if mask.sum()==0: continue
            md = mask.to(probs.device)
            res[key]['tot'] += int(mask.sum().item())
            res[key]['ok']  += int((pred[md]==yb[md]).sum().item())
            res[key]['ents']+= ent[md].tolist()
            res[key]['probs'].append(probs[md].cpu())
            res[key]['labels'].append(yb[md].cpu())
    out={}
    for key in ['ID','OOD']:
        tot=res[key]['tot']
        if tot==0:
            out[key]={'acc':float('nan'),'entropy':float('nan'),'ece':float('nan')}
            continue
        probs_cat=torch.cat(res[key]['probs'],0) if res[key]['probs'] else torch.empty(0)
        labels_cat=torch.cat(res[key]['labels'],0) if res[key]['labels'] else torch.empty(0, dtype=torch.long)
        out[key]={'acc': res[key]['ok']/tot,
                  'entropy': float(torch.tensor(res[key]['ents']).mean()) if res[key]['ents'] else float('nan'),
                  'ece': ece_score(probs_cat, labels_cat, 15) if probs_cat.numel()>0 else float('nan')}
    return out

@torch.no_grad()
def per_path_diagnostics(model, dl):
    model.eval()
    tot=0; ok_f=0; ok_c=0
    probs_f_all=[]; probs_c_all=[]; labels_all=[]
    for xb, xcfb, yb in dl:
        xb, xcfb, yb = xb.to(device), xcfb.to(device), yb.to(device)
        _, _, lf, lc, _, _ = model(xb, xcfb)
        pf = F.softmax(lf, dim=-1); pc = F.softmax(lc, dim=-1)
        ok_f += (pf.argmax(-1)==yb).sum().item()
        ok_c += (pc.argmax(-1)==yb).sum().item()
        tot += yb.size(0)
        probs_f_all.append(pf.cpu()); probs_c_all.append(pc.cpu()); labels_all.append(yb.cpu())
    pf_cat = torch.cat(probs_f_all, 0); pc_cat = torch.cat(probs_c_all, 0); y_cat = torch.cat(labels_all, 0)
    return {"acc_f": ok_f/tot, "acc_c": ok_c/tot, "ece_f": ece_score(pf_cat, y_cat, 15), "ece_c": ece_score(pc_cat, y_cat, 15)}

@torch.no_grad()
def average_gate(model, dl):
    model.eval(); g_id=0.0;n_id=0; g_ood=0.0;n_ood=0
    for xb, xcfb, _ in dl:
        xb, xcfb = xb.to(device), xcfb.to(device)
        _, g, _, _, _, _ = model(xb, xcfb)
        sm = surprise_mask(xb.cpu())
        if (~sm).sum()>0:
            idx=(~sm).to(g.device); g_id += float(g[idx].mean().item())*int((~sm).sum().item()); n_id += int((~sm).sum().item())
        if sm.sum()>0:
            idx=sm.to(g.device); g_ood += float(g[idx].mean().item())*int(sm.sum().item()); n_ood += int(sm.sum().item())
    return (g_id/max(n_id,1), g_ood/max(n_ood,1))

# ------------------------- Plot Helpers --------------------------------------

def ensure_outdir(path="outputs"):
    os.makedirs(path, exist_ok=True)
    return path

def reliability_bins(probs, labels, n_bins=15):
    conf, pred = probs.max(-1)
    acc = (pred==labels).float()
    bins = torch.linspace(0,1,n_bins+1)
    mids = 0.5*(bins[:-1]+bins[1:])
    acc_bin=[]; conf_bin=[]; count=[]
    for i in range(n_bins):
        m = (conf>bins[i]) & (conf<=bins[i+1])
        if m.any():
            acc_bin.append(acc[m].mean().item()); conf_bin.append(conf[m].mean().item()); count.append(m.sum().item())
        else:
            acc_bin.append(np.nan); conf_bin.append(np.nan); count.append(0)
    return mids.numpy(), np.array(acc_bin), np.array(conf_bin), np.array(count)

def plot_reliability(ax, probs, labels, title):
    mids, acc_bin, conf_bin, count = reliability_bins(probs, labels, n_bins=15)
    mask = ~np.isnan(acc_bin)
    ax.plot([0,1],[0,1], linestyle="--", linewidth=1)
    ax.bar(mids[mask], (conf_bin[mask]-acc_bin[mask]), width=1/15, alpha=0.35, label="gap (conf-acc)")
    ax2 = ax.twinx()
    if count[mask].sum() > 0:
        ax2.bar(mids[mask], count[mask]/max(count[mask].max(),1), width=1/15, alpha=0.12, color="gray")
    ax2.set_ylim(0,1); ax2.set_yticks([])
    ax.plot(mids[mask], acc_bin[mask], marker="o", linewidth=1.5, label="accuracy")
    ax.plot(mids[mask], conf_bin[mask], marker="o", linewidth=1.5, label="confidence")
    ax.set_title(title); ax.set_xlabel("confidence"); ax.set_ylabel("value"); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.legend(loc="lower right", fontsize=8)

@torch.no_grad()
def compute_mixer_stats(logits, lf, lc, sm_mask=None):
    fused = F.softmax(logits, dim=-1)
    p_f = F.softmax(lf / CFG.expert_temp, dim=-1)
    p_c = F.softmax(lc / CFG.expert_temp, dim=-1)
    eps=1e-9
    Hf = -(p_f * (p_f.clamp_min(eps)).log()).sum(-1)
    Hc = -(p_c * (p_c.clamp_min(eps)).log()).sum(-1)
    Hfused = -(fused * (fused.clamp_min(eps)).log()).sum(-1)
    conf = torch.stack([-Hf, -Hc], dim=-1) / CFG.tau_conf
    a = F.softmax(conf, dim=-1)[:, 1]
    m = 0.5*(p_f+p_c)
    js = 0.5*((p_f*(p_f/(m+eps)).clamp_min(eps).log()).sum(-1) + (p_c*(p_c/(m+eps)).clamp_min(eps).log()).sum(-1))
    beta_js = torch.sigmoid(3.0 * (js - js.mean()))
    conf_brake = torch.sigmoid(CFG.fused_conf_slope * (Hfused - CFG.fused_conf_pivot))
    beta = torch.clamp(CFG.disagree_mix_lambda * beta_js * conf_brake, 0.0, 1.0)
    if sm_mask is not None:
        sm = sm_mask.to(beta.device)
        if sm.any():
            beta = beta.clone(); beta[sm] = torch.minimum(beta[sm], torch.full_like(beta[sm], 0.15))
            a = a.clone(); a[sm] = 0.5 * a[sm]
    return {"beta": beta, "a": a, "js": js, "Hf": Hf, "Hc": Hc, "Hfused": Hfused}

@torch.no_grad()
def collect_full_outputs(model, dl, scaler_id, scaler_ood):
    model.eval()
    P=[]; Y=[]; YH=[]; C=[]; H=[]; SM=[]; G=[]; B=[]; A=[]; JS=[]
    for xb, xcfb, yb in dl:
        xb, xcfb, yb = xb.to(device), xcfb.to(device), yb.to(device)
        logits, g, lf, lc, _, _ = model(xb, xcfb)
        sm = surprise_mask(xb.cpu())
        probs = anchored_mix(logits, lf, lc, sm_mask=sm)
        if (scaler_id is not None) or (scaler_ood is not None):
            probs = apply_split_scaling(probs, xb, scaler_id, scaler_ood)
        probs = apply_split_entropy_floor(probs, xb)
        md = compute_mixer_stats(logits, lf, lc, sm_mask=sm)
        pred = probs.argmax(-1)
        conf = probs.max(-1).values
        ent  = -(probs * (probs.clamp_min(1e-9)).log()).sum(-1)
        P.append(probs.cpu()); Y.append(yb.cpu()); YH.append(pred.cpu())
        C.append(conf.cpu()); H.append(ent.cpu()); SM.append(sm.cpu())
        G.append(g.cpu()); B.append(md["beta"].cpu()); A.append(md["a"].cpu()); JS.append(md["js"].cpu())
    return {
        "probs":torch.cat(P,0), "y":torch.cat(Y,0), "yhat":torch.cat(YH,0),
        "conf":torch.cat(C,0), "H":torch.cat(H,0), "sm":torch.cat(SM,0).bool(),
        "g":torch.cat(G,0), "beta":torch.cat(B,0), "a":torch.cat(A,0), "js":torch.cat(JS,0)
    }

# --- Extra: Risk–Coverage & Accuracy–Confidence ---
def risk_coverage(probs, labels, n=50):
    conf = probs.max(-1).values
    pred = probs.argmax(-1)
    correct = (pred == labels).float()
    ths = torch.linspace(0, 1, n)
    cov, risk = [], []
    for t in ths:
        m = conf >= t
        c = m.float().mean().item()
        r = (1 - correct[m]).mean().item() if m.any() else float("nan")
        cov.append(c); risk.append(r)
    return ths.cpu().numpy(), np.array(cov), np.array(risk)

def acc_vs_conf(probs, labels, n=50):
    conf = probs.max(-1).values; pred = probs.argmax(-1)
    ths = torch.linspace(0,1,n)
    accs=[]
    for t in ths:
        m = conf >= t
        accs.append( (pred[m]==labels[m]).float().mean().item() if m.any() else np.nan )
    return ths.cpu().numpy(), np.array(accs)

# --- Compact 2×3 panel (drop-in) ---
def _rel_bins_local(probs, labels, n_bins=15):
    if isinstance(probs, np.ndarray): probs = torch.tensor(probs)
    if isinstance(labels, np.ndarray): labels = torch.tensor(labels, dtype=torch.long)
    conf, pred = probs.max(-1); acc = (pred == labels).float()
    bins = torch.linspace(0, 1, n_bins + 1); mids = 0.5 * (bins[:-1] + bins[1:])
    acc_bin, conf_bin, count = [], [], []
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.any():
            acc_bin.append(acc[m].mean().item()); conf_bin.append(conf[m].mean().item()); count.append(int(m.sum().item()))
        else:
            acc_bin.append(np.nan); conf_bin.append(np.nan); count.append(0)
    return mids.numpy(), np.array(acc_bin), np.array(conf_bin), np.array(count)

def make_summary_panel_from_pack(pack, itos, save_dir="outputs", show=True):
    os.makedirs(save_dir, exist_ok=True)
    P, Y, YH = pack["probs"], pack["y"], pack["yhat"]
    C, H, G, B, JS, SM = pack["conf"], pack["H"], pack["g"], pack["beta"], pack["js"], pack["sm"].bool()
    id_mask, ood_mask = (~SM), (SM)

    # 2×3 panel
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=120)

    # (1) Reliability ID
    mids, acc_bin, conf_bin, count = _rel_bins_local(P[id_mask], Y[id_mask], n_bins=15)
    m = ~np.isnan(acc_bin); ax = axes[0,0]
    ax.plot([0,1],[0,1], ls="--", lw=1)
    ax.bar(mids[m], (conf_bin[m]-acc_bin[m]), width=1/15, alpha=0.35, label="gap (conf-acc)")
    ax2 = ax.twinx()
    if count[m].sum() > 0:
        ax2.bar(mids[m], count[m]/max(count[m].max(),1), width=1/15, alpha=0.12, color="gray")
    ax2.set_ylim(0,1); ax2.set_yticks([])
    ax.plot(mids[m], acc_bin[m], marker="o", lw=1.5, label="accuracy")
    ax.plot(mids[m], conf_bin[m], marker="o", lw=1.5, label="confidence")
    ax.set_title("Reliability (ID)"); ax.set_xlabel("confidence"); ax.set_ylabel("value")
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.legend(loc="lower right", fontsize=8)

    # (2) Reliability OOD
    mids, acc_bin, conf_bin, count = _rel_bins_local(P[ood_mask], Y[ood_mask], n_bins=15)
    m = ~np.isnan(acc_bin); ax = axes[0,1]
    ax.plot([0,1],[0,1], ls="--", lw=1)
    ax.bar(mids[m], (conf_bin[m]-acc_bin[m]), width=1/15, alpha=0.35, label="gap (conf-acc)")
    ax2 = ax.twinx()
    if count[m].sum() > 0:
        ax2.bar(mids[m], count[m]/max(count[m].max(),1), width=1/15, alpha=0.12, color="gray")
    ax2.set_ylim(0,1); ax2.set_yticks([])
    ax.plot(mids[m], acc_bin[m], marker="o", lw=1.5, label="accuracy")
    ax.plot(mids[m], conf_bin[m], marker="o", lw=1.5, label="confidence")
    ax.set_title("Reliability (OOD)"); ax.set_xlabel("confidence"); ax.set_ylabel("value")
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.legend(loc="lower right", fontsize=8)

    # (3) Confidence hist
    ax = axes[0,2]
    ax.hist(C[id_mask].cpu().numpy(), bins=30, alpha=0.8, label="ID")
    ax.hist(C[ood_mask].cpu().numpy(), bins=30, alpha=0.6, label="OOD")
    ax.set_title("Confidence histogram"); ax.legend()

    # (4) Entropy hist
    ax = axes[1,0]
    ax.hist(H[id_mask].cpu().numpy(), bins=30, alpha=0.8, label="ID")
    ax.hist(H[ood_mask].cpu().numpy(), bins=30, alpha=0.6, label="OOD")
    ax.set_title("Entropy histogram"); ax.legend()

    # (5) Gate g hist
    ax = axes[1,1]
    ax.hist(G[id_mask].cpu().numpy(), bins=30, alpha=0.85, label="ID")
    ax.hist(G[ood_mask].cpu().numpy(), bins=30, alpha=0.65, label="OOD")
    ax.set_title("Gate g"); ax.legend()

    # (6) β vs JS scatter (OOD)
    ax = axes[1,2]
    ood_idx = ood_mask.cpu().numpy()
    correct = (YH[ood_mask] == Y[ood_mask]).cpu().numpy()
    sc = ax.scatter(JS[ood_mask].cpu().numpy(), B[ood_mask].cpu().numpy(),
                    c=correct.astype(float), cmap='coolwarm', alpha=0.75)
    ax.axhline(0.15, ls="--", c="gray", label="β cap")
    ax.set_xlabel("JS(p_f || p_c)"); ax.set_ylabel("β (override)"); ax.set_title("OOD: β vs JS")
    ax.legend(); fig.colorbar(sc, ax=ax, label="correct (1) / wrong (0)")

    plt.tight_layout()
    panel_path = os.path.join(save_dir, "panel_summary.png")
    fig.savefig(panel_path); 
    if show: plt.show()
    plt.close(fig)

    # Confusion matrix
    num_classes = P.shape[1]
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t,p in zip(Y.cpu().numpy(), YH.cpu().numpy()):
        cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(6,6), dpi=120)
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Confusion matrix (fused)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
    ax.set_xticklabels([t for t in itos[:num_classes]], rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels([t for t in itos[:num_classes]], fontsize=7)
    plt.tight_layout()
    cm_path = os.path.join(save_dir, "panel_confusion.png")
    fig.savefig(cm_path); 
    if show: plt.show()
    plt.close(fig)
    print(f"Saved {panel_path} and {cm_path}")

# ------------------------- Main ----------------------------------------------

def main(show_plots=True, outdir="outputs"):
    outdir = ensure_outdir(outdir)

    print("Training baseline…")
    baseline = train(Baseline().to(device), dual_flag=False)

    print("\nTraining dual (x-attn, CF warm-up, OOD rehearsal with fused supervision)…")
    dual = train(DualPathXAttn().to(device), dual_flag=True)

    print("\nFitting split calibrators (ID: VectorScaling; OOD: Temp-only)…")
    P_id, Y_id, P_ood_nat, Y_ood_nat = split_val_probs(dual, val_dl)
    scaler_id  = fit_vs_from_probs(P_id,  Y_id,  vocab_size, l2=CFG.vs_l2)
    xb_ood_syn, xcfb_ood_syn, yb_ood_syn = synth_ood_batch(batch_size=512)
    with torch.no_grad():
        logits_syn, _, lf_syn, lc_syn, _, _ = dual(xb_ood_syn.to(device), xcfb_ood_syn.to(device))
        sm_syn = surprise_mask(xb_ood_syn.cpu())
        P_ood_syn = anchored_mix(logits_syn, lf_syn, lc_syn, sm_mask=sm_syn).detach().cpu()
    scaler_ood = fit_temp_from_probs(P_ood_syn, yb_ood_syn.cpu(), vocab_size)

    print("\n=== Aggregate Test ===")
    base_all = evaluate(baseline, test_dl, dual_flag=False, scaler_id=None, scaler_ood=None, use_split_floor=False)
    dual_all = evaluate(dual, test_dl, dual_flag=True, scaler_id=scaler_id, scaler_ood=scaler_ood, use_split_floor=True)
    print("Baseline:", base_all)
    print("DualPath (v16):", dual_all)

    print("\n=== ID vs OOD (surprise) ===")
    base_split = eval_split(baseline, test_dl, dual_flag=False, scaler_id=None, scaler_ood=None, use_split_floor=False)
    dual_split = eval_split(dual, test_dl, dual_flag=True, scaler_id=scaler_id, scaler_ood=scaler_ood, use_split_floor=True)
    print("Baseline:", base_split)
    print("DualPath (v16):", dual_split)

    pp = per_path_diagnostics(dual, test_dl)
    print("\nPer-path diagnostics (test):", pp)
    gid, good = average_gate(dual, test_dl)
    print(f"Avg gate g (ID):  {gid:.3f}")
    print(f"Avg gate g (OOD): {good:.3f}")

    # --------- Save metrics JSON + CSV ---------
    metrics = {
        "aggregate": {"baseline": base_all, "dual": dual_all},
        "split": {"baseline": base_split, "dual": dual_split},
        "per_path": pp,
        "gate_avg": {"id": gid, "ood": good}
    }
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(outdir, "quick_stats.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["split","acc","entropy","ece","g_avg"])
        w.writerow(["ID",  dual_split["ID"]["acc"],  dual_split["ID"]["entropy"],  dual_split["ID"]["ece"],  gid ])
        w.writerow(["OOD", dual_split["OOD"]["acc"], dual_split["OOD"]["entropy"], dual_split["OOD"]["ece"], good])
    print(f"Saved metrics to {os.path.join(outdir, 'metrics.json')} and quick_stats.csv")

    # --------- Plots (individual) ---------
    print("\nCollecting outputs for plots…")
    pack = collect_full_outputs(dual, test_dl, scaler_id, scaler_ood)
    sm = pack["sm"].numpy()
    id_mask  = ~sm; ood_mask = sm

    # Reliability
    fig, axes = plt.subplots(1,2, figsize=(10,4), dpi=120)
    plot_reliability(axes[0], pack["probs"][id_mask],  pack["y"][id_mask],  "Reliability (ID)")
    plot_reliability(axes[1], pack["probs"][ood_mask], pack["y"][ood_mask], "Reliability (OOD)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "reliability_id_ood.png"))
    if show_plots: plt.show(); plt.close()

    # Confidence & Entropy hist
    P  = pack["probs"]; Y = pack["y"]; YH = pack["yhat"]
    C  = pack["conf"].numpy(); H = pack["H"].numpy()
    G  = pack["g"].numpy();    B = pack["beta"].numpy(); JS = pack["js"].numpy()

    fig, axes = plt.subplots(1,2, figsize=(10,4), dpi=120)
    axes[0].hist(C[id_mask], bins=30, alpha=0.8, label="ID")
    axes[0].hist(C[ood_mask], bins=30, alpha=0.6, label="OOD")
    axes[0].set_title("Confidence histogram"); axes[0].legend()
    axes[1].hist(H[id_mask], bins=30, alpha=0.8, label="ID")
    axes[1].hist(H[ood_mask], bins=30, alpha=0.6, label="OOD")
    axes[1].set_title("Entropy histogram"); axes[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "hist_conf_entropy.png"))
    if show_plots: plt.show(); plt.close()

    # Gate g, β, JS hists
    fig, axes = plt.subplots(1,3, figsize=(15,4), dpi=120)
    axes[0].hist(G[id_mask], bins=30, alpha=0.85, label="ID"); axes[0].hist(G[ood_mask], bins=30, alpha=0.65, label="OOD")
    axes[0].set_title("Gate g"); axes[0].legend()
    axes[1].hist(B[id_mask], bins=30, alpha=0.85, label="ID"); axes[1].hist(B[ood_mask], bins=30, alpha=0.65, label="OOD")
    axes[1].set_title("Mixer override β"); axes[1].legend()
    axes[2].hist(JS[id_mask], bins=30, alpha=0.85, label="ID"); axes[2].hist(JS[ood_mask], bins=30, alpha=0.65, label="OOD")
    axes[2].set_title("Expert JS divergence"); axes[2].legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "hists_g_beta_js.png"))
    if show_plots: plt.show(); plt.close()

    # OOD scatter β vs JS
    ood_idx = np.where(ood_mask)[0]
    correct = (YH[ood_idx]==Y[ood_idx]).numpy()
    fig, ax = plt.subplots(figsize=(6,5), dpi=120)
    ax.scatter(JS[ood_idx][~correct], B[ood_idx][~correct], s=12, alpha=0.7, label="wrong")
    ax.scatter(JS[ood_idx][correct],   B[ood_idx][correct],   s=12, alpha=0.7, label="correct")
    ax.axhline(0.15, ls="--", c="gray", label="β cap")
    ax.set_xlabel("JS(p_f || p_c)"); ax.set_ylabel("β (expert override)"); ax.set_title("OOD: β vs JS")
    ax.legend(); plt.tight_layout(); plt.savefig(os.path.join(outdir, "ood_scatter_beta_vs_js.png"))
    if show_plots: plt.show(); plt.close()

    # Confusion matrix
    num_classes = P.shape[1]
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t,p in zip(Y.numpy(), YH.numpy()): cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(5.5,5), dpi=120)
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Confusion matrix (fused)"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
    ax.set_xticklabels([t for t in itos[:num_classes]], rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels([t for t in itos[:num_classes]], fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "confusion_matrix.png"))
    if show_plots: plt.show(); plt.close()

    # Risk–Coverage + Accuracy–Confidence
    fig, ax = plt.subplots(1,2, figsize=(10,4), dpi=120)
    ths, cov_id, risk_id = risk_coverage(P[~pack["sm"]], Y[~pack["sm"]])
    ths, cov_ood, risk_ood = risk_coverage(P[ pack["sm"]], Y[ pack["sm"]])
    ax[0].plot(cov_id, risk_id, label="ID"); ax[0].plot(cov_ood, risk_ood, label="OOD")
    ax[0].set_xlabel("coverage"); ax[0].set_ylabel("risk (1-acc)"); ax[0].set_title("Risk–Coverage"); ax[0].legend()

    ths, acc_id = acc_vs_conf(P[~pack["sm"]], Y[~pack["sm"]])
    ths, acc_ood = acc_vs_conf(P[ pack["sm"]], Y[ pack["sm"]])
    ax[1].plot(ths, acc_id, label="ID"); ax[1].plot(ths, acc_ood, label="OOD")
    ax[1].set_xlabel("confidence threshold"); ax[1].set_ylabel("accuracy"); ax[1].set_title("Accuracy vs Confidence"); ax[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "risk_coverage_and_acc_vs_conf.png"))
    if show_plots: plt.show(); plt.close()

    # Compact 2×3 panel + confusion matrix panel
    make_summary_panel_from_pack(pack, itos, save_dir=outdir, show=show_plots)

    print(f"\nPlots saved in: {outdir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true", help="Save plots without displaying them.")
    parser.add_argument("--outdir", type=str, default="outputs", help="Directory to save plots.")
    args = parser.parse_args()
    main(show_plots=not args.no_show, outdir=args.outdir)

