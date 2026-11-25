/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   This software is distributed under the GNU General Public License.

   fix becfield/kk: EGCNN Born effective charges (EFMLP method 2) on the GPU.  See fix_becfield_kokkos.h.

   Network (BEC_PRED/<system>/model.py, EquivarBECModel, exported by tools/export_egcnn.py):
     node features x (N x 768) = 64x0e + 64x1o + 64x1e + 64x2e, e3nn layout [block][channel u][m]
     edges src -> dst with vector v = r_dst - r_src (here: receiver i, neighbour j, v = x_i - x_j)
     per layer: w_e = cut(r) * (W2 silu(W1 rbf(r) + b1) + b2)   (17 paths x 64 channels)
                agg_i = sum_e TP_uvu(x_src, Y(v/|v|), w_e);  x += agg L
     Z* = x Mout (output linear + CartesianTensor), minus the mean over all atoms.
   The network itself is in becfield_core.cuh (shared with the Python module); this file builds the
   edges from the LAMMPS Kokkos neighbour list and applies forces / accumulates the polarization.
------------------------------------------------------------------------- */

#ifdef SYMMETRIX_CUDA

#include "fix_becfield_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "input.h"
#include "kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "pair.h"
#include "update.h"
#include "variable.h"

#include "becfield_core.cuh"


#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>

using namespace LAMMPS_NS;
using namespace FixConst;
using AT = ArrayTypes<LMPDeviceType>;

namespace becfield {

// edges from the LAMMPS Kokkos full list: one warp per receiver atom, order-preserving compaction
__device__ __forceinline__ bool edge_ok(AT::t_kkfloat_1d_3_lr x, AT::t_neighbors_2d neighbors, int i, int jj,
                                        double rc2, int &j, double &vx, double &vy, double &vz)
{
  j = neighbors(i, jj) & NEIGHMASK;
  vx = x(i, 0) - x(j, 0);    // r_dst - r_src (receiver i, sender j)
  vy = x(i, 1) - x(j, 1);
  vz = x(i, 2) - x(j, 2);
  return vx * vx + vy * vy + vz * vz < rc2;
}

__global__ void k_count(int n, AT::t_kkfloat_1d_3_lr x, AT::t_int_1d numneigh,
                        AT::t_neighbors_2d neighbors, double rc2, int *cnt)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (i > n) return;
  if (i == n) {
    if (lane == 0) cnt[n] = 0;
    return;
  }
  int c = 0;
  const int nn = numneigh(i);
  for (int jj = lane; jj < nn; jj += 32) {
    int j;
    double vx, vy, vz;
    c += edge_ok(x, neighbors, i, jj, rc2, j, vx, vy, vz);
  }
  for (int o = 16; o > 0; o >>= 1) c += __shfl_xor_sync(~0u, c, o);
  if (lane == 0) cnt[i] = c;
}

__global__ void k_fill(int n, AT::t_kkfloat_1d_3_lr x, AT::t_int_1d numneigh, AT::t_neighbors_2d neighbors,
                       Kokkos::View<int *> owner, const int *rowptr, double rc2, int *src, double *vec, int *err)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (i >= n) return;
  int base = rowptr[i];
  const int nn = numneigh(i);
  for (int j0 = 0; j0 < nn; j0 += 32) {
    const int jj = j0 + lane;
    int j = 0;
    double vx = 0, vy = 0, vz = 0;
    const bool ok = jj < nn && edge_ok(x, neighbors, i, jj, rc2, j, vx, vy, vz);
    const unsigned m = __ballot_sync(~0u, ok);
    if (ok) {
      const int e = base + __popc(m & ((1u << lane) - 1u));
      const int s = owner(j);
      if (s < 0 || s >= n) atomicOr(err, 1);
      src[e] = s < 0 ? 0 : s;
      vec[3L * e] = vx;
      vec[3L * e + 1] = vy;
      vec[3L * e + 2] = vz;
    }
    base += __popc(m);
  }
}

struct Box {
  double eps[3], boxlo[3], hinv[6], cell[3][3], cellavg[3][3];
  int have_prev;
};

// forces F_i[b] += sum_a eps_a Z_i[a,b]; polarization dP (trapezoid, minimum image, mean cell), state by tag
__global__ void k_final(int n, const double *Zd, Box b, AT::t_kkfloat_1d_3_lr x, AT::t_tagint_1d tag,
                        AT::t_kkacc_1d_3 f, double *prevfrac, double *prevZ, double *dP)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  double p[3] = {0, 0, 0};
  if (i < n) {
    double Z[9];
    for (int k = 0; k < 9; ++k) Z[k] = Zd[(long) i * 9 + k];
    for (int bb = 0; bb < 3; ++bb)
      f(i, bb) += b.eps[0] * Z[bb] + b.eps[1] * Z[3 + bb] + b.eps[2] * Z[6 + bb];
    const double dx = x(i, 0) - b.boxlo[0], dy = x(i, 1) - b.boxlo[1], dz = x(i, 2) - b.boxlo[2];
    const double s[3] = {b.hinv[0] * dx + b.hinv[5] * dy + b.hinv[4] * dz, b.hinv[1] * dy + b.hinv[3] * dz,
                         b.hinv[2] * dz};
    const long t = tag(i) - 1;
    if (b.have_prev) {
      double ds[3], dr[3];
      for (int k = 0; k < 3; ++k) {
        ds[k] = s[k] - prevfrac[t * 3 + k];
        ds[k] -= rint(ds[k]);
      }
      for (int bb = 0; bb < 3; ++bb)
        dr[bb] = ds[0] * b.cellavg[0][bb] + ds[1] * b.cellavg[1][bb] + ds[2] * b.cellavg[2][bb];
      for (int a = 0; a < 3; ++a)
        for (int bb = 0; bb < 3; ++bb) p[a] += 0.5 * (Z[a * 3 + bb] + prevZ[t * 9 + a * 3 + bb]) * dr[bb];
    }
    for (int k = 0; k < 3; ++k) prevfrac[t * 3 + k] = s[k];
    for (int k = 0; k < 9; ++k) prevZ[t * 9 + k] = Z[k];
  }
  for (int a = 0; a < 3; ++a)
    for (int o = 16; o > 0; o >>= 1) p[a] += __shfl_xor_sync(~0u, p[a], o);
  if ((threadIdx.x & 31) == 0)
    for (int a = 0; a < 3; ++a) atomicAdd(&dP[a], p[a]);
}

// LAMMPS side: Kokkos neighbour list -> core -> forces and polarization
struct EvaluatorBase {
  std::unique_ptr<CoreBase> core;
  DevBuf<double> prevfrac, prevZ, dP;
  double rc = 0;
  EvaluatorBase(CoreBase *c) : core(c), rc(c->rc) { dP.need(3); }
  long edges() const { return core->edges(); }

  void compute(int n, int natoms, AT::t_kkfloat_1d_3_lr xv, AT::t_int_1d type, AT::t_tagint_1d tag,
               AT::t_kkacc_1d_3 f, AT::t_int_1d numneigh, AT::t_neighbors_2d neighbors,
               Kokkos::View<int *> owner, const Box &box, double dPh[3])
  {
    cudaStream_t stream = Kokkos::DefaultExecutionSpace().cuda_stream();
    core->set_stream(stream);
    const int TB = 128;
    const int nw = ((n + 1) * 32 + TB - 1) / TB;    // one warp per atom
    int *cnt = core->count_buffer(n);
    k_count<<<nw, TB, 0, stream>>>(n, xv, numneigh, neighbors, rc * rc, cnt);
    core->scan(n);
    k_fill<<<nw, TB, 0, stream>>>(n, xv, numneigh, neighbors, owner, core->rowptr(), rc * rc, core->src(),
                                  core->vec(), core->errflag());
    const double *Zd = core->run(n, type.data());
    prevfrac.need((size_t) natoms * 3);
    prevZ.need((size_t) natoms * 9);
    CK(cudaMemsetAsync(dP.p, 0, 3 * sizeof(double), stream));
    k_final<<<(n + TB - 1) / TB, TB, 0, stream>>>(n, Zd, box, xv, tag, f, prevfrac.p, prevZ.p, dP.p);
    int herr = 0;
    CK(cudaMemcpyAsync(dPh, dP.p, 3 * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CK(cudaMemcpyAsync(&herr, core->errflag(), sizeof(int), cudaMemcpyDeviceToHost, stream));
    core->mark("output");
    CK(cudaStreamSynchronize(stream));
    CK(cudaGetLastError());
    core->accumulate_profile();
    if (herr) throw std::runtime_error("a neighbour has no owned atom (atom map / ghost inconsistency)");
  }

  void get_z(std::vector<double> &z, int natoms)
  {
    z.resize((size_t) natoms * 9);
    CK(cudaMemcpy(z.data(), prevZ.p, z.size() * sizeof(double), cudaMemcpyDeviceToHost));
  }
};

}    // namespace becfield

/* ---------------------------------------------------------------------- */

static const char *const PT[] = {
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",  "Cl",
    "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se",
    "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb",
    "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
    "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At",
    "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm"};

FixBECFieldKokkos::FixBECFieldKokkos(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = Device;
  datamask_read = X_MASK | F_MASK | TYPE_MASK | TAG_MASK;
  datamask_modify = F_MASK;

  restart_global = 1;
  vector_flag = 1;
  size_vector = 6;
  global_freq = 1;
  extvector = 0;
  thermo_energy = 0;
  energy_global_flag = 0;
  virial_global_flag = 0;

  if (!lmp->kokkos || lmp->kokkos->ngpus < 1)
    error->all(FLERR, "fix becfield/kk requires KOKKOS with a CUDA GPU ('-k on g 1 -sf kk')");
  const int ntypes = atom->ntypes;
  if (narg < 4 + ntypes)
    error->all(FLERR, "Usage: fix ID group becfield/kk model.json elem1 ... elemN(ntypes) efield Ex Ey Ez "
                      "[p0 Px Py Pz] [precision float32|float64] [radial spline|exact] [grid N] [chunk Nedges] "
                      "[zfile file every] "
                      "[timing yes|no]");
  model_file = arg[3];
  for (int t = 0; t < ntypes; ++t) elements.emplace_back(arg[4 + t]);
  bool have_field = false;
  int iarg = 4 + ntypes;
  while (iarg < narg) {
    const std::string kw = arg[iarg];
    auto need = [&](int k) {
      if (iarg + k >= narg) error->all(FLERR, "fix becfield/kk: missing value(s) after {}", kw);
    };
    if (kw == "efield") {
      need(3);
      for (int d = 0; d < 3; ++d) {
        efield_str[d] = arg[iarg + 1 + d];
        if (!utils::strmatch(efield_str[d], "^v_"))
          efield_val[d] = utils::numeric(FLERR, efield_str[d], false, lmp);
      }
      have_field = true;
      iarg += 4;
    } else if (kw == "p0") {
      need(3);
      for (int d = 0; d < 3; ++d) P[d] = utils::numeric(FLERR, arg[iarg + 1 + d], false, lmp);
      iarg += 4;
    } else if (kw == "precision") {
      need(1);
      const std::string v = arg[iarg + 1];
      if (v == "float32") use_double = false;
      else if (v == "float64") use_double = true;
      else error->all(FLERR, "fix becfield/kk precision must be float32 or float64");
      iarg += 2;
    } else if (kw == "radial") {
      need(1);
      const std::string v = arg[iarg + 1];
      if (v == "spline") radial_exact = false;
      else if (v == "exact") radial_exact = true;
      else error->all(FLERR, "fix becfield/kk radial must be spline or exact");
      iarg += 2;
    } else if (kw == "grid") {
      need(1);
      grid = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      if (grid < 64) error->all(FLERR, "fix becfield/kk grid must be >= 64");
      iarg += 2;
    } else if (kw == "chunk") {
      need(1);
      max_edges = utils::bnumeric(FLERR, arg[iarg + 1], false, lmp);
      if (max_edges < 1000) error->all(FLERR, "fix becfield/kk chunk must be >= 1000 edges");
      iarg += 2;
    } else if (kw == "zfile") {
      need(2);
      zfile = arg[iarg + 1];
      zevery = utils::inumeric(FLERR, arg[iarg + 2], false, lmp);
      if (zevery < 1) error->all(FLERR, "fix becfield/kk zfile interval must be >= 1");
      iarg += 3;
    } else if (kw == "timing") {
      need(1);
      timing = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else {
      error->all(FLERR, "fix becfield/kk: unknown keyword {}", kw);
    }
  }
  if (!have_field) error->all(FLERR, "fix becfield/kk requires the efield keyword");

  std::vector<int> Z;
  for (auto &e : elements) {
    int z = -1;
    for (int k = 0; k < 100; ++k)
      if (e == PT[k]) z = k + 1;
    if (z < 0) error->all(FLERR, "fix becfield/kk: unknown element {}", e);
    Z.push_back(z);
  }
  try {
    std::ifstream in(model_file);
    if (!in) error->all(FLERR, "fix becfield/kk: cannot open {}", model_file);
    auto j = nlohmann::json::parse(in);
    ev = std::make_unique<becfield::EvaluatorBase>(
        becfield::make_core(j, Z, use_double, max_edges, radial_exact, grid));
  } catch (std::exception &e) {
    error->all(FLERR, "fix becfield/kk: {}", e.what());
  }
  if (comm->me == 0)
    utils::logmesg(lmp, "fix becfield/kk: EGCNN {} ({}, radial {}), cutoff {} A, elements:{}\n", model_file,
                   use_double ? "float64" : "float32",
                   radial_exact ? "exact (GEMM)" : fmt::format("spline, {} intervals", grid), ev->rc, [&] {
                     std::string s;
                     for (auto &e : elements) s += " " + e;
                     return s;
                   }());
}

FixBECFieldKokkos::~FixBECFieldKokkos() = default;

int FixBECFieldKokkos::setmask()
{
  return POST_FORCE;    // nothing during minimize: no field force, no polarization update
}

void FixBECFieldKokkos::init()
{
  if (comm->nprocs > 1) error->all(FLERR, "fix becfield/kk supports a single MPI rank only");
  if (igroup != group->find("all")) error->all(FLERR, "fix becfield/kk must be applied to group all");
  if (atom->map_user == atom->MAP_NONE)
    error->all(FLERR, "fix becfield/kk requires 'atom_modify map [yes|array|hash]'");
  if (!atom->tag_enable || atom->natoms != (bigint) atom->map_tag_max)
    error->all(FLERR, "fix becfield/kk requires consecutive atom IDs 1..N");
  if (!force->pair || force->pair->cutforce < ev->rc)
    error->all(FLERR, "fix becfield/kk needs a pair style with cutoff >= {} A (the EGCNN cutoff)", ev->rc);
  for (int d = 0; d < 3; ++d) {
    efield_var[d] = -1;
    if (utils::strmatch(efield_str[d], "^v_")) {
      efield_var[d] = input->variable->find(efield_str[d].c_str() + 2);
      if (efield_var[d] < 0)
        error->all(FLERR, "Variable {} for fix becfield/kk does not exist", efield_str[d].substr(2));
      if (!input->variable->equalstyle(efield_var[d]))
        error->all(FLERR, "Variable {} for fix becfield/kk must be equal-style", efield_str[d].substr(2));
    }
  }
  auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
  request->set_kokkos_host(0);
  request->set_kokkos_device(1);
}

void FixBECFieldKokkos::init_list(int, NeighList *ptr)
{
  list = ptr;
}

void FixBECFieldKokkos::setup(int vflag)
{
  post_force(vflag);
}

void FixBECFieldKokkos::post_force(int)
{
  const auto t0 = std::chrono::steady_clock::now();
  for (int d = 0; d < 3; ++d) {
    field[d] = efield_var[d] >= 0 ? input->variable->compute_equal(efield_var[d]) : efield_val[d];
    if (!std::isfinite(field[d])) error->all(FLERR, "fix becfield/kk efield component {} is not finite", d + 1);
  }
  atomKK->sync(execution_space, datamask_read);
  auto x = atomKK->k_x.view<LMPDeviceType>();
  auto f = atomKK->k_f.view<LMPDeviceType>();
  auto type = atomKK->k_type.view<LMPDeviceType>();
  auto tag = atomKK->k_tag.view<LMPDeviceType>();
  auto k_list = static_cast<NeighListKokkos<LMPDeviceType> *>(list);
  const int nlocal = atom->nlocal, nall = atom->nlocal + atom->nghost;

  if ((int) owner.extent(0) < nall) Kokkos::realloc(owner, nall);
  {
    auto own = owner;
    auto map_style = atom->map_style;
    auto k_map_array = atomKK->k_map_array;
    auto k_map_hash = atomKK->k_map_hash;
    k_map_array.template sync<LMPDeviceType>();
    Kokkos::parallel_for(
        "FixBECFieldKokkos::owner", nall, KOKKOS_LAMBDA(const int i) {
          own(i) = AtomKokkos::map_kokkos<LMPDeviceType>(tag(i), map_style, k_map_array, k_map_hash);
        });
  }

  becfield::Box b{};
  const double *h = domain->h, *hi = domain->h_inv;
  for (int d = 0; d < 3; ++d) {
    b.eps[d] = field[d];
    b.boxlo[d] = domain->boxlo[d];
  }
  for (int k = 0; k < 6; ++k) b.hinv[k] = hi[k];
  auto cellof = [](const double *hh, double c[3][3]) {
    const double m[3][3] = {{hh[0], 0, 0}, {hh[5], hh[1], 0}, {hh[4], hh[3], hh[2]}};
    std::memcpy(c, m, sizeof(m));
  };
  double cnow[3][3], cprev[3][3];
  cellof(h, cnow);
  cellof(hprev, cprev);
  std::memcpy(b.cell, cnow, sizeof(cnow));
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) b.cellavg[r][c] = 0.5 * (cnow[r][c] + cprev[r][c]);
  b.have_prev = have_prev ? 1 : 0;

  double dP[3] = {0, 0, 0};
  try {
    ev->compute(nlocal, (int) atom->natoms, x, type, tag, f, k_list->d_numneigh, k_list->d_neighbors, owner, b,
                dP);
  } catch (std::exception &e) {
    error->one(FLERR, "fix becfield/kk: {}", e.what());
  }
  for (int d = 0; d < 3; ++d) P[d] += dP[d];
  for (int k = 0; k < 6; ++k) hprev[k] = h[k];
  have_prev = true;
  atomKK->modified(execution_space, datamask_modify);

  if (zevery > 0 && update->ntimestep % zevery == 0) write_z();
  if (timing) {
    t_total += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    ++n_calls;
  }
}

void FixBECFieldKokkos::write_z()
{
  std::vector<double> z;
  ev->get_z(z, (int) atom->natoms);
  if (comm->me != 0) return;
  FILE *fp = fopen(zfile.c_str(), "a");
  if (!fp) error->one(FLERR, "fix becfield/kk: cannot open {}", zfile);
  utils::print(fp, "# step {} natoms {} field {:.10g} {:.10g} {:.10g} P {:.12g} {:.12g} {:.12g}\n",
             update->ntimestep, atom->natoms, field[0], field[1], field[2], P[0], P[1], P[2]);
  for (bigint t = 0; t < atom->natoms; ++t) {
    utils::print(fp, "{}", t + 1);
    for (int k = 0; k < 9; ++k) utils::print(fp, " {:.9e}", z[t * 9 + k]);
    fputc('\n', fp);
  }
  fclose(fp);
}

void FixBECFieldKokkos::post_run()
{
  if (timing && n_calls > 0 && comm->me == 0)
    utils::logmesg(lmp, "fix becfield/kk: {} calls, {:.3f} ms per call (incl. sync), {} edges\n", n_calls,
                   1e3 * t_total / n_calls, ev->edges());
  t_total = 0;
  n_calls = 0;
}

double FixBECFieldKokkos::compute_vector(int n)
{
  return n < 3 ? P[n] : field[n - 3];
}

void FixBECFieldKokkos::write_restart(FILE *fp)
{
  if (comm->me == 0) {
    int size = 3 * sizeof(double);
    fwrite(&size, sizeof(int), 1, fp);
    fwrite(P, sizeof(double), 3, fp);
  }
}

void FixBECFieldKokkos::restart(char *buf)
{
  auto *list_ = (double *) buf;
  for (int d = 0; d < 3; ++d) P[d] = list_[d];
  have_prev = false;
}

#endif
