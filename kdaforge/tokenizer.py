# -*- coding: utf-8 -*-
"""A minimal, correct character-level tokenizer (reference).

The focus of this project is the attention / MoE / distillation stack, not
subword modeling, so the tokenizer is deliberately simple: every Unicode
character maps to an id, with a small set of control tokens. ``encode`` /
``decode`` round-trip exactly, and long documents simply become long token
sequences -- which is exactly what the 1M-context linear attention is built
for.
"""

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL = [PAD, BOS, EOS, UNK]


class Tokenizer:
    def __init__(self, vocab=None):
        self.special = SPECIAL
        self.stoi = {s: i for i, s in enumerate(SPECIAL)}
        self.itos = list(SPECIAL)
        self._next = len(SPECIAL)

    def _id_for(self, ch):
        if ch not in self.stoi:
            self.stoi[ch] = self._next
            self.itos.append(ch)
            self._next += 1
        return self.stoi[ch]

    def encode(self, text, add_bos=False, add_eos=False):
        ids = [self._id_for(ch) for ch in text]
        if add_bos:
            ids = [self.stoi[BOS]] + ids
        if add_eos:
            ids = ids + [self.stoi[EOS]]
        return ids

    def decode(self, ids):
        out = []
        for i in ids:
            tok = self.itos[i] if 0 <= i < len(self.itos) else UNK
            if tok in SPECIAL:
                continue
            out.append(tok)
        return "".join(out)

    @property
    def vocab_size(self):
        return len(self.itos)

    def pad(self, ids, length):
        return ids[:length] + [self.stoi[PAD]] * max(0, length - len(ids))


def _stable_token_ids(text, n, seed="tok"):
    """Deterministic pseudo token ids used by benchmarks that need synthetic
    sequences without instantiating a full vocabulary."""
    from .utils.random import stable_ints
    return stable_ints(seed + text, n, 0, 256)
