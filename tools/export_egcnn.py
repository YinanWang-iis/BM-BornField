#!/usr/bin/env python
"""Export an EFMLP EGCNN (EquivarBECModel, BEC_PRED/<system>) to the JSON read by LAMMPS `fix becfield/kk`.

    python export_egcnn.py BEC_PRED/BaTiO3 -o egcnn_bto.json

Needs the environment the model was trained in (e3nn 0.4.4 + torch_geometric), CPU only.
Everything e3nn-specific is turned into plain arrays (float64):
  x0[Z]         initial node features (embedding -> embed_linear), Z = 0..99
  layers[k]     W1 (64x32), b1, W2 (1088x64), b2 (radial MLP), L (768x768): x += agg @ L
  Mout          768x9: Z*[a,b] (row-major) = x @ Mout   (output_linear followed by CartesianTensor('ij'))
  tp            17 paths: (l1, l2, l3) blocks and the coefficient tensor C[a,b,c] incl. e3nn path weights
  sh            spherical harmonics (component normalisation) as homogeneous polynomials of u = v/(|v|+1e-8)
The tp/sh coefficients only depend on the architecture; the CUDA kernels have them compiled in
(egcnn_generated.hpp) and the fix checks that the file agrees.
"""
import argparse, json, os, sys, itertools
import numpy as np
import torch

torch.serialization.add_safe_globals([slice])
torch.set_default_dtype(torch.float64)

ARCH = dict(num_layers=6, feature_size=64, num_radial=32, edge_sh_lmax=2, cutoff=4.5, embedding_dim=32,
            equivariant_cart=True)


def load(model_dir):
    sys.path.insert(0, os.path.abspath(model_dir))
    from model import EquivarBECModel
    net = EquivarBECModel(**ARCH)
    sd = torch.load(os.path.join(model_dir, "trained_model.pth"), map_location="cpu")
    net.load_state_dict({k: v.double() for k, v in sd.items()})
    return net.double().eval()


def offsets(irreps):
    o, out = 0, []
    for mul, ir in irreps:
        out.append((o, mul, ir.l))
        o += mul * (2 * ir.l + 1)
    return out


@torch.no_grad()
def tp_paths(tp):
    oi, osh, oo = offsets(tp.irreps_in1), offsets(tp.irreps_in2), offsets(tp.irreps_out)
    paths = []
    for p, ins in enumerate(tp.instructions):
        (s1, _, l1), (s2, _, l2), (s3, _, l3) = oi[ins.i_in1], osh[ins.i_in2], oo[ins.i_out]
        C = np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1))
        w = torch.zeros(1, tp.weight_numel); w[0, p * 64] = 1.0
        for a in range(2 * l1 + 1):
            for b in range(2 * l2 + 1):
                x = torch.zeros(1, tp.irreps_in1.dim); x[0, s1 + a] = 1
                y = torch.zeros(1, tp.irreps_in2.dim); y[0, s2 + b] = 1
                out = tp(x, y, w)[0]
                C[a, b] = out[s3:s3 + 2 * l3 + 1].numpy()
                rest = out.clone(); rest[s3:s3 + 2 * l3 + 1] = 0
                assert rest.abs().max() < 1e-12
        paths.append(dict(i1=ins.i_in1, i2=ins.i_in2, io=ins.i_out, l1=l1, l2=l2, l3=l3, C=C.tolist()))
    return paths


def sh_polys(net):
    from e3nn.o3 import spherical_harmonics
    rng = np.random.default_rng(0)
    v = torch.tensor(rng.normal(size=(200, 3)))
    Y = spherical_harmonics(net.irreps_sh, v, normalize=False, normalization="component").numpy()
    mons = [(0, 0, 0)] + [m for d in (1, 2) for m in itertools.product(range(d + 1), repeat=3) if sum(m) == d]
    A = np.stack([np.prod(v.numpy() ** np.array(m), axis=1) for m in mons], 1)
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    coef[np.abs(coef) < 1e-12] = 0
    assert np.abs(A @ coef - Y).max() < 1e-10
    return dict(monomials=mons, coef=coef.T.tolist())     # Y[k] = sum_m coef[k][m] * u^mons[m]


@torch.no_grad()
def export(model_dir):
    net = load(model_dir)
    I = torch.eye(net.irreps_node.dim)
    out = dict(format="egcnn-bec-1", source=os.path.abspath(model_dir), arch=ARCH,
               irreps_node=str(net.irreps_node), irreps_sh=str(net.irreps_sh),
               rbf_means=net.rbf.means.tolist(), rbf_gamma=float(net.rbf.gamma),
               x0=net.embed_linear(net.embedding.weight).tolist(),
               Mout=(net.output_linear(I) @ net.cart_Q).tolist(),
               tp=tp_paths(net.conv_layers[0].tp), sh=sh_polys(net), layers=[])
    for conv, lin in zip(net.conv_layers, net.res_linears):
        m = conv.radial_mlp
        out["layers"].append(dict(W1=m[0].weight.tolist(), b1=m[0].bias.tolist(),
                                  W2=m[2].weight.tolist(), b2=m[2].bias.tolist(), L=lin(I).tolist()))
    for conv in net.conv_layers[1:]:          # identical CG paths in every layer
        assert json.dumps(tp_paths(conv.tp)) == json.dumps(out["tp"])
    return out, net


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args()
    out, _ = export(a.model_dir)
    with open(a.output, "w") as f:
        json.dump(out, f)
    print("written", a.output, f"({os.path.getsize(a.output) / 1e6:.1f} MB)")
