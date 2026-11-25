# compare sp_pair_<tag> (egcnn LAMMPS build) vs field/tests/lammps/sp (original sm_120 build) and the Python prototype
import numpy as np, re, sys
tag = sys.argv[1] if len(sys.argv) > 1 else 'sm120'
D = '/homenfsxx3/ynwang/symmetrix_test/field/tests/lammps'
def dump(p): return np.loadtxt(p, skiprows=9)[:, 1:]
def res(log):
    for l in open(log):
        m = re.match(r'RESULT \S+ (\S+) (\S+) (\S+) (\S+)', l)
        if m: r = (float(m.group(1)), np.array([float(m.group(i)) for i in (2, 3, 4)]))
    return r
py = np.load(f'{D}/py_ref.npy', allow_pickle=True).item()
N = {'bto40': 40, 'bto320': 320}
print(f'struct dtype field | {tag}-sm120: dE meV/atom  dF max meV/A  dP max eA | {tag}-python: dE  dF  dP | P(z)')
for s in N:
    for t in ['f64', 'f32', 'mix']:
        for fi in range(4):
            k = f'{s}_{t}_f{fi}'
            E, P = res(f'sp_pair_{tag}/{k}.log'); F = dump(f'sp_pair_{tag}/{k}.dump')
            Er, Pr = res(f'{D}/sp/{k}.log'); Fr = dump(f'{D}/sp/{k}.dump'); R = py[k]
            print(f"{s:7s} {t:4s} f{fi} | {(E-Er)/N[s]*1e3:10.2e} {np.abs(F-Fr).max()*1e3:10.2e} {np.abs(P-Pr).max():9.2e} |"
                  f" {(E-R['E'])/N[s]*1e3:10.2e} {np.abs(F-R['F']).max()*1e3:10.2e} {np.abs(P-R['P']).max():9.2e} | {P[2]:.6f}")
