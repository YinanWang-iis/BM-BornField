"""Plain-torch reference of the exported EGCNN JSON (no e3nn). Used to validate export and the CUDA fix.
    ref = EGCNNRef("egcnn_bto.json"); Z = ref.becs(numbers, positions, cell)   # (N,3,3), ASR applied
Edge convention as in the training code: message src -> dst with vector r_dst - r_src."""
import json
import numpy as np
import torch


class EGCNNRef:
    def __init__(self, path, dtype=torch.float64, device="cpu"):
        d = json.load(open(path))
        t = lambda x: torch.tensor(x, dtype=dtype, device=device)
        self.d, self.t = d, t
        self.cut = d["arch"]["cutoff"]
        self.means, self.gamma = t(d["rbf_means"]), d["rbf_gamma"]
        self.x0, self.Mout = t(d["x0"]), t(d["Mout"])
        self.layers = [{k: t(v) for k, v in L.items()} for L in d["layers"]]
        F = d["arch"]["feature_size"]
        self.F = F
        self.blk = [(0, 0), (F, 1), (4 * F, 1), (7 * F, 2)]       # node irreps 0e 1o 1e 2e: (offset, l)
        self.shblk = [(0, 0), (1, 1), (4, 2)]
        self.paths = [(p["i1"], p["i2"], p["io"], t(p["C"])) for p in d["tp"]]
        self.mons = d["sh"]["monomials"]; self.shc = t(d["sh"]["coef"])

    def sh(self, u):
        A = torch.stack([u[:, 0] ** m[0] * u[:, 1] ** m[1] * u[:, 2] ** m[2] for m in self.mons], 1)
        return A @ self.shc.T

    def forward(self, numbers, src, dst, vec):
        F, N = self.F, len(numbers)
        dist = vec.norm(dim=-1)
        rbf = torch.exp(-self.gamma * (dist[:, None] - self.means) ** 2)
        x_ = dist / self.cut
        cut = torch.where(x_ < 1, torch.exp(-x_ ** 2 / (1 - x_ ** 2 + 1e-8)), torch.zeros_like(x_))
        Y = self.sh(vec / (dist[:, None] + 1e-8))
        x = self.x0[numbers]
        for L in self.layers:
            h = torch.nn.functional.silu(rbf @ L["W1"].T + L["b1"])
            w = (h @ L["W2"].T + L["b2"]) * cut[:, None]                   # (E, 17*F)
            xs = x[src]
            msg = torch.zeros_like(xs)
            for p, (i1, i2, io, C) in enumerate(self.paths):
                (o1, l1), (o2, l2), (o3, l3) = self.blk[i1], self.shblk[i2], self.blk[io]
                a = xs[:, o1:o1 + F * (2 * l1 + 1)].reshape(-1, F, 2 * l1 + 1)
                y = Y[:, o2:o2 + 2 * l2 + 1]
                m = torch.einsum("eua,eb,abc->euc", a, y, C) * w[:, p * F:(p + 1) * F, None]
                msg[:, o3:o3 + F * (2 * l3 + 1)] += m.reshape(-1, F * (2 * l3 + 1))
            agg = torch.zeros_like(x).index_add_(0, dst, msg)
            x = x + agg @ L["L"]
        Z = x @ self.Mout
        return Z - Z.mean(0, keepdim=True)

    def becs(self, numbers, positions, cell, cutoff=None):
        from ase.neighborlist import neighbor_list
        from ase import Atoms
        i, j, D = neighbor_list("ijD", Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True),
                                cutoff or self.cut)
        t = lambda a, dt=None: torch.as_tensor(np.asarray(a), dtype=dt) if dt else torch.as_tensor(np.asarray(a))
        Z = self.forward(t(numbers, torch.long), t(i, torch.long), t(j, torch.long), self.t(D))
        return Z.numpy().reshape(-1, 3, 3)
