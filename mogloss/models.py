"""Models. Parameter names match the released checkpoints.

MotionEncoder   one token per keypoint frame (207-d) -> Linear -> + learned position -> pre-norm Transformer -> LayerNorm
MMM             encoder + learned [MASK] vector + linear head that regresses the masked frames (stage 1)
JointModel      the stage-1 model plus pseudo-gloss embeddings; one Transformer reads [units] + [frames] (stage 2)
Translator      encoder -> Linear -> T5-base encoder -> T5-base decoder
Classifier      encoder -> masked mean pooling -> LayerNorm / Dropout / Linear (isolated sign recognition)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from .features import FRAME_DIM

PAD, MASK, UNK = 0, 1, 2
MAX_UNITS = 40


class MotionEncoder(nn.Module):
    def __init__(self, din=FRAME_DIM, d=1024, layers=16, heads=16, max_len=1032, drop=0.1):
        super().__init__()
        self.inp = nn.Linear(din, d)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)

    def encode(self, x, m):
        """x (B, L, din) standardised frames, m (B, L) True for valid frames -> (B, L, d)."""
        h = self.inp(x) + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        return self.norm(self.enc(h, src_key_padding_mask=~m))


class MMM(nn.Module):
    """Masked motion modeling: masked frames are replaced by a learned vector after the input layer."""

    def __init__(self, din=FRAME_DIM, d=1024, layers=16, heads=16, max_len=1032):
        super().__init__()
        self.enc = MotionEncoder(din, d, layers, heads, max_len)
        self.mask_tok = nn.Parameter(torch.randn(d) * 0.02)
        self.head = nn.Linear(d, din)

    def forward(self, x, m, sel):
        """x (B, L, din) standardised, m valid frames, sel masked frames -> reconstruction (B, L, din)."""
        e = self.enc
        h = e.inp(x)
        h = torch.where(sel[..., None], self.mask_tok.to(h.dtype), h) + e.pos(torch.arange(x.shape[1], device=x.device))[None]
        return self.head(e.norm(e.enc(h, src_key_padding_mask=~m)))


class JointModel(nn.Module):
    """Joint motion and pseudo-gloss masked modeling. Units and frames share one Transformer; a modality embedding
    marks the two streams. Unit logits are tied to the unit embedding."""

    def __init__(self, vocab_size, din=FRAME_DIM, d=1024, layers=16, heads=16, max_len=1032, mu=None, sd=None):
        super().__init__()
        self.m = MMM(din, d, layers, heads, max_len)
        self.register_buffer('mu', torch.zeros(din) if mu is None else torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer('sd', torch.ones(din) if sd is None else torch.as_tensor(sd, dtype=torch.float32))
        self.tok = nn.Embedding(vocab_size, d)
        nn.init.normal_(self.tok.weight, std=0.02)
        self.tpos = nn.Embedding(MAX_UNITS + 1, d)
        self.mod = nn.Embedding(2, d)
        nn.init.zeros_(self.mod.weight)
        self.tnorm = nn.LayerNorm(d)

    def forward(self, x, m, u, msel, drop_text=None):
        """x raw frames (B, L, din), m valid, u unit ids (B, T) with MASK at masked units, msel masked frames,
        drop_text (B,) hides all units. Returns frame reconstruction (standardised) and unit logits."""
        e = self.m.enc
        z = (x - self.mu) / self.sd
        hm = e.inp(z)
        hm = torch.where(msel[..., None], self.m.mask_tok.to(hm.dtype), hm)
        hm = hm + e.pos(torch.arange(x.shape[1], device=x.device))[None] + self.mod.weight[1]
        tm = u != PAD
        if drop_text is not None:
            tm = tm & ~drop_text[:, None]
        ht = self.tok(u) + self.tpos(torch.arange(u.shape[1], device=u.device).clamp(max=MAX_UNITS))[None] + self.mod.weight[0]
        h = e.norm(e.enc(torch.cat([ht.to(hm.dtype), hm], 1), src_key_padding_mask=~torch.cat([tm, m], 1)))
        T = u.shape[1]
        return self.m.head(h[:, T:]), self.tnorm(h[:, :T]) @ self.tok.weight.T


def joint_loss(model, x, m, u, g, p_text_mode=0.5, p_drop_text=0.1):
    """Each sample is either a text sample (15-50 % of its units masked, cross-entropy) or a motion sample
    (30-50 % of its frames masked, L1 on standardised frames; with probability 0.1 its units are hidden)."""
    B = len(x)
    dev = x.device
    tmode = torch.rand(B, device=dev, generator=g) < p_text_mode
    tm = u != PAD
    r = torch.rand(u.shape, device=dev, generator=g)
    frac = 0.15 + 0.35 * torch.rand(B, 1, device=dev, generator=g)
    tsel = (r < frac) & tm & tmode[:, None]
    first = torch.argmax((tm & tmode[:, None]).float() * (1 + torch.rand(u.shape, device=dev, generator=g)), 1)
    tsel[torch.arange(B, device=dev), first] |= tmode & tm.any(1)  # at least one masked unit
    rm = torch.rand(m.shape, device=dev, generator=g)
    fm = 0.3 + 0.2 * torch.rand(B, 1, device=dev, generator=g)
    msel = (rm < fm) & m & ~tmode[:, None]
    drop = (~tmode) & (torch.rand(B, device=dev, generator=g) < p_drop_text)
    ui = torch.where(tsel, torch.full_like(u, MASK), u)
    ym, yt = model(x, m, ui, msel, drop)
    lt = F.cross_entropy(yt[tsel].float(), u[tsel]) if tsel.any() else yt.sum() * 0
    zs = (x - model.mu) / model.sd
    lm = ((ym.float() - zs).abs().mean(-1) * msel).sum() / msel.sum().clamp_min(1)
    acc = (yt[tsel].argmax(-1) == u[tsel]).float().mean().item() if tsel.any() else 0.0
    return lt + lm, lt.item(), lm.item(), acc


class Translator(nn.Module):
    def __init__(self, din=FRAME_DIM, d=1024, layers=16, heads=16, max_len=1032, mu=None, sd=None, t5='t5-base'):
        super().__init__()
        self.enc = MotionEncoder(din, d, layers, heads, max_len)
        self.register_buffer('mu', torch.zeros(din) if mu is None else torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer('sd', torch.ones(din) if sd is None else torch.as_tensor(sd, dtype=torch.float32))
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5)
        self.proj = nn.Linear(d, self.t5.config.d_model)

    def encode(self, x, m):
        h = self.proj(self.enc.encode((x - self.mu) / self.sd, m))
        return BaseModelOutput(last_hidden_state=self.t5.encoder(inputs_embeds=h, attention_mask=m.long()).last_hidden_state)

    def forward(self, x, m, labels):
        return self.t5(encoder_outputs=self.encode(x, m), attention_mask=m.long(), labels=labels).loss

    @torch.no_grad()
    def generate(self, x, m, tokenizer, beams=4, max_len=64):
        out = self.t5.generate(encoder_outputs=self.encode(x, m), attention_mask=m.long(), num_beams=beams, max_new_tokens=max_len)
        return tokenizer.batch_decode(out, skip_special_tokens=True)


class Classifier(nn.Module):
    def __init__(self, n_cls, din=FRAME_DIM, d=1024, layers=16, heads=16, max_len=1032, drop=0.2):
        super().__init__()
        self.enc = MotionEncoder(din, d, layers, heads, max_len)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(drop), nn.Linear(d, n_cls))

    def forward(self, x, m):
        h = self.enc.encode(x, m)
        w = m.float().unsqueeze(-1)
        return self.head((h * w).sum(1) / w.sum(1).clamp_min(1))


def load_encoder_checkpoint(path):
    """Encoder checkpoint -> (encoder state dict with 'enc.' stripped, mu, sd, config)."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    enc = {k[4:]: v for k, v in ck['model'].items() if k.startswith('enc.')}
    return enc, np.asarray(ck['mu'], np.float32), np.asarray(ck['sd'], np.float32), ck.get('config', {})
