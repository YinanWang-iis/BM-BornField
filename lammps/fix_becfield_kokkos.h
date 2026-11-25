/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   This software is distributed under the GNU General Public License.

   fix ID all becfield/kk <model.json> <elem1> ... <elemN> efield Ex Ey Ez [keywords]
     Field-induced forces from an EGCNN Born-effective-charge network (EFMLP method 2), on the GPU:
       Z*_i   = EGCNN(R) with the acoustic sum rule (mean over all atoms subtracted)
       F_i[b] += sum_a eps_a Z*_i[a,b]     (no energy, no virial)
       dP_a  += sum_i 1/2 (Z*_i(t) + Z*_i(t-dt))[a,b] dr_i[b]   (trapezoid, minimum image, mean cell)
     global vector: [1..3] accumulated dP (e*Angstrom), [4..6] current field (V/Angstrom)
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
#ifdef SYMMETRIX_CUDA
FixStyle(becfield/kk,FixBECFieldKokkos);
FixStyle(becfield/kk/device,FixBECFieldKokkos);
#endif
// clang-format on
#else

#ifndef LMP_FIX_BECFIELD_KOKKOS_H
#define LMP_FIX_BECFIELD_KOKKOS_H
#ifdef SYMMETRIX_CUDA

#include "fix.h"
#include "kokkos_type.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace becfield {
struct EvaluatorBase;
}

namespace LAMMPS_NS {

class FixBECFieldKokkos : public Fix {
 public:
  FixBECFieldKokkos(class LAMMPS *, int, char **);
  ~FixBECFieldKokkos() override;
  int setmask() override;
  void init() override;
  void init_list(int, class NeighList *) override;
  void setup(int) override;
  void post_force(int) override;
  void post_run() override;
  double compute_vector(int) override;
  void write_restart(FILE *) override;
  void restart(char *) override;

 protected:
  std::string model_file;
  std::vector<std::string> elements;
  std::array<std::string, 3> efield_str;
  std::array<int, 3> efield_var{-1, -1, -1};
  std::array<double, 3> efield_val{0, 0, 0};
  std::array<double, 3> field{0, 0, 0};
  double P[3] = {0, 0, 0};
  double hprev[6] = {0, 0, 0, 0, 0, 0};
  bool have_prev = false;
  bool use_double = false;
  long max_edges = 1L << 18;
  bool radial_exact = false;
  int grid = 1024;
  std::string zfile;
  int zevery = 0;
  bool timing = false;
  double t_total = 0;
  long n_calls = 0;
  Kokkos::View<int *> owner;
  class NeighList *list = nullptr;
  std::unique_ptr<becfield::EvaluatorBase> ev;
  void write_z();
};

}    // namespace LAMMPS_NS

#endif
#endif
#endif
