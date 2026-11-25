# BM-BornField

GPU (Kokkos/CUDA) LAMMPS interface for **field-induced forces from a Born-effective-charge network**.

At every MD step the LAMMPS fix `fix becfield/kk` does the following:

- predicts the Born effective charges Z\*ᵢ of all atoms with an equivariant graph convolutional neural network (EGCNN, the BM1 architecture of [Kutana *et al.*, *Sci. Rep.* 2025](https://doi.org/10.1038/s41598-025-01250-5));
- adds the field force Fᵢ = ε·Z\*ᵢ;
- accumulates the polarization change ΔP.

It is meant to be used together with a zero-field machine-learning potential. In our workflow this is MACE through
[symmetrix-MACE-Field](https://github.com/YinanWang-iis/symmetrix-MACE-Field)
(`pair_style symmetrix/mace/foundation/float32`).

Compared with the previous implementation (`fix external pf/callback` + e3nn model in Python), the Born-charge part
of an MD step is about **30× faster**:

| System | Python callback (e3nn, GPU) | `fix becfield/kk` |
|---|---|---|
| 2,560-atom BaTiO3 | 147 ms | 4.6 ms |
| 8,960-atom bilayer hBN | 445 ms | 15.5 ms |

Both timings are per step, float32, measured on an RTX 5090 that was shared with another job. The predicted Z\*
agree with the Python model to ≤ 7×10⁻⁶ e per component.

## Physics

The fix implements the following, with ε the uniform external field in V/Å:

- **Z\***: Z\*ᵢ[a,b] = EGCNN(R)ᵢ, with the acoustic sum rule applied (the mean over all atoms is subtracted).
  - a is the field / polarization direction and b is the displacement direction.
  - The 9 components are stored row-major, the same convention as VASP.
- **Force**: Fᵢ[b] += Σₐ εₐ Z\*ᵢ[a,b]. Because of the sum rule, Σᵢ Fᵢ = 0.
  - This force has **no energy and no virial**. That is the defining approximation of this "BEC × field" approach.
- **Polarization**: ΔPₐ += Σᵢ ½ (Z\*ᵢ(t) + Z\*ᵢ(t−Δt))[a,b] Δrᵢ[b], a trapezoidal accumulation.
  - Δr is the minimum-image displacement, taken from fractional coordinates and the mean of the two cells.
  - The unit is e·Å, for the whole simulation box.

## Requirements and build

You need:

- LAMMPS with KOKKOS (CUDA). Tested with stable 30Sep2026 Update1 and Kokkos 5.2.1.
- `pair_symmetrix` from [symmetrix-MACE-Field](https://github.com/YinanWang-iis/symmetrix-MACE-Field), installed into the
  same LAMMPS tree. The fix uses its CMake block for the `SYMMETRIX_CUDA` definition, cuBLAS linking and the bundled
  `nlohmann_json.hpp`.

Build steps:

```bash
bash symmetrix-MACE-Field/pair_symmetrix/install.sh path/to/lammps
bash BM-BornField/lammps/install.sh path/to/lammps          # links fix_becfield* into src/ and src/KOKKOS/
cmake -S path/to/lammps/cmake -B build -D PKG_KOKKOS=ON -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ARCH_<GPU>=ON \
      -D SYMMETRIX_KOKKOS=ON -D CMAKE_CXX_STANDARD=20 -D BUILD_MPI=OFF ...   # see cluster/build_lammps.sh
```

Tested builds:

| Target | Kokkos / CUDA arch | GPUs it runs on |
|---|---|---|
| `sm120` | `Kokkos_ARCH_BLACKWELL120`, CUDA arch 120 | RTX 5090, RTX PRO 6000/5000 Blackwell |
| `sm80` | `Kokkos_ARCH_AMPERE80`, CUDA arch 80 | A100, RTX 3090, RTX 4090, RTX 6000 Ada |

`cluster/` contains the exact scripts used on our cluster:

- `build_lammps.sh`;
- `env.sh`, which picks the build from the GPU's compute capability;
- `bin/lmp`, a wrapper that does the same for the LAMMPS executable.

## Model export

The network weights are converted once, in the environment the model was trained in (e3nn 0.4.4, torch_geometric; CPU
is enough):

```bash
python tools/export_egcnn.py path/to/BEC_PRED/BaTiO3 -o egcnn_bto.json
```

What the export does:

- **Format**: e3nn modules become plain float64 arrays: embedding, radial MLPs, the 17 `uvu` tensor-product paths
  (including e3nn normalisation), the per-layer linears, and the output linear with `CartesianTensor('ij')`.
- **Accuracy**: the export reproduces the e3nn model to 5×10⁻¹⁵ e.
- **Reference implementation**: `tools/egcnn_ref.py` is a plain-torch reference of the exported file.

**Supported architecture.** Only `EquivarBECModel(num_layers=6, feature_size=64, num_radial=32,
edge_sh_lmax=2, cutoff=4.5, embedding_dim=32, equivariant_cart=True)` is supported. The Clebsch–Gordan coefficients are
compiled into the kernels: `lammps/egcnn_generated.hpp` is produced by `tools/gen_kernels.py`, and the fix checks that
the model file matches it. Other architectures need a regenerated header and a rebuild.

Exported model files and training data are not distributed in this repository.

## Usage

```
atom_modify  map yes
pair_style   symmetrix/mace/foundation/float32
pair_coeff   * * mace_bto.json Ba Ti O
variable     EFX equal 0.0
variable     EFY equal 0.0
variable     EFZ equal "0.03*sin(2*PI*step/20000)"     # V/Angstrom, re-evaluated every step
fix          ef all becfield egcnn_bto.json Ba Ti O efield v_EFX v_EFY v_EFZ
thermo_style custom step temp pe f_ef[1] f_ef[2] f_ef[3]
```

Run with `lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.xxx`.

The general syntax is:

```
fix ID all becfield/kk model.json elem_type1 ... elem_typeN efield Ex Ey Ez [keywords]
```

- `Ex Ey Ez` are numbers or `v_name` equal-style variables, in V/Å.
- The element symbols are listed in LAMMPS atom-type order.

| Keyword | Default | Meaning |
|---|---|---|
| `p0 Px Py Pz` | 0 0 0 | initial accumulated polarization (e·Å), for restarts from your own checkpoints |
| `precision float32\|float64` | float32 | precision of the network evaluation |
| `radial spline\|exact` | spline | radial weights from a cubic-Hermite table (error ≈ 10⁻⁸ e), or from the MLP via cuBLAS GEMMs (reference) |
| `grid N` | 1024 | number of spline intervals on [0, r_c] |
| `chunk N` | 262144 | `radial exact` only: maximum number of edges per batch, to limit memory |
| `zfile file every` | – | append Z\* of all atoms (by atom ID, 9 components) every `every` steps |
| `timing yes\|no` | no | print the mean time per call at the end of each run |

**Outputs.** The fix provides a global vector:

- `f_ef[1..3]` is the accumulated ΔP in e·Å;
- `f_ef[4..6]` is the current field in V/Å.

**Restarts.**

- ΔP is stored in restart files. After `read_restart`, define the fix again with the same ID to restore it.
- Alternatively, start from `write_data` output and pass the saved value with `p0`.

**Minimization.** During `minimize` the fix does nothing: it adds no field force and does not update ΔP, so it can stay
defined.

At the setup of the next `run`, the displacement made by the relaxation is added to ΔP with the trapezoidal rule. To
leave it out, define the fix after the minimization.

**Python driver.** When LAMMPS is driven from Python, call `lmp.finalize()` after `lmp.close()`. Otherwise Kokkos aborts
at interpreter exit with `cudaErrorCudartUnloading`. This also happens without this fix.

## Validation

All scripts are in `tests/` and the logs in `tests/logs/`. There are two references:

- the original e3nn model in float32;
- the exported network evaluated in float64.

**Z\* and field forces.** Structures: BaTiO3 (40 and 2,560 atoms) and hBN (36 and 8,960 atoms). The field forces are
compared with `pair_style zero`, at ε = (0, 0, 0.03) and (0.02, −0.01, 0.03) V/Å.

| Precision | max \|Z\* − e3nn float32\| | max \|Z\* − reference float64\| | Field-force error |
|---|---|---|---|
| float32 | ≤ 7×10⁻⁶ e | ≤ 3.4×10⁻⁶ e | ≤ 1×10⁻⁴ meV/Å |
| float64 | – | ≤ 1×10⁻⁸ e | ≤ 3×10⁻⁷ meV/Å |

The e3nn model's own float32 error is 0.5–6×10⁻⁶ e. With `radial exact` in float64 the kernels agree with the
reference to 5×10⁻¹⁰ e, the print precision. In all cases |ΣF_field| < 10⁻⁹ meV/Å.

**Polarization.** Setup: 200-step NVE runs under a time-dependent field, for 320-atom BaTiO3 and 324-atom hBN. ΔP
agrees with a Python replay of the original accumulation within 1.2×10⁻⁶ e·Å. For comparison, |ΔP| ≈ 90 e·Å in the
BaTiO3 run.

**Restarts.** Continuing from `read_restart`, and from `write_data` + `p0`, reproduces ΔP within the float32 noise of
the trajectory (≤ 1×10⁻⁵ e·Å).

**Minimization.** `minimize` (cg) runs with the fix defined.

**Coexistence with symmetrix.** The `pair_symmetrix` results in the same LAMMPS build are unchanged: fp64 is bitwise
identical to the build without this fix.

**Hardware.** The tests were run on an RTX 5090 (sm120 build) and an RTX 3090 (sm80 build).

## Limitations

- **Single process**:
  - one MPI rank and one GPU;
  - group `all` only;
  - atom IDs must be consecutive 1..N;
  - `atom_modify map yes` is required.
- **Neighbour list**: the pair style cutoff must be ≥ 4.5 Å, because the fix reuses its full Kokkos neighbour list.
- **Architecture**: only the network architecture listed above is supported.
- **Thermodynamics**: there is no energy or virial contribution, so the pressure contains no field term.
  - ΔP under a varying cell (NPT) uses the same mean-cell formula as the original code, but it has only been validated
    in NVE.
- **No CPU version**: plain `fix becfield` stops with an error asking for `-sf kk`.

## How to cite

If you use this code, please cite the paper that introduced the Born-effective-charge network:

> A. Kutana, K. Shimizu, S. Watanabe, R. Asahi, "Representing Born effective charges with equivariant graph
> convolutional neural networks", *Scientific Reports* **15**, 16719 (2025).
> https://doi.org/10.1038/s41598-025-01250-5

```bibtex
@article{Kutana2025BEC,
  author  = {Kutana, Alex and Shimizu, Koji and Watanabe, Satoshi and Asahi, Ryoji},
  title   = {Representing Born effective charges with equivariant graph convolutional neural networks},
  journal = {Scientific Reports},
  volume  = {15},
  pages   = {16719},
  year    = {2025},
  doi     = {10.1038/s41598-025-01250-5}
}
```

Also cite the software it builds on:

- **LAMMPS**: A. P. Thompson *et al.*, "LAMMPS – a flexible simulation tool for particle-based materials modeling at
  the atomic, meso, and continuum scales", *Comput. Phys. Commun.* **271**, 108171 (2022).
  https://doi.org/10.1016/j.cpc.2021.108171
- **KOKKOS package of LAMMPS**: A. Johansson *et al.*, "LAMMPS-KOKKOS: Performance Portable Molecular Dynamics Across
  Exascale Architectures", *Proc. SC '25 Workshops*, 1217–1232 (2025). https://doi.org/10.1145/3731599.3767498
- **Kokkos**: C. R. Trott *et al.*, "Kokkos 3: Programming Model Extensions for the Exascale Era", *IEEE Trans.
  Parallel Distrib. Syst.* **33**, 805–817 (2022). https://doi.org/10.1109/TPDS.2021.3097283
- **e3nn**, in which the network is defined and trained, and whose conventions the export reproduces: M. Geiger and
  T. Smidt, "e3nn: Euclidean Neural Networks", arXiv:2207.09453 (2022).
- **When used with MACE / symmetrix for the zero-field potential**:
  - I. Batatia *et al.*, "MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force
    Fields", *Advances in Neural Information Processing Systems* **35** (2022);
  - symmetrix, https://github.com/wcwitt/symmetrix.

## Acknowledgements

The implementation strategy follows [symmetrix](https://github.com/wcwitt/symmetrix) by C. Witt *et al.*:

- hand-written CUDA kernels that take the neighbour list directly from LAMMPS Kokkos;
- radial functions tabulated as splines.

The ghost-to-owner atom mapping is adapted from symmetrix's `pair_symmetrix` (GPLv2).

## License

GPL-2.0, the same as LAMMPS, which `fix becfield` is part of when compiled. See `LICENSE`.
