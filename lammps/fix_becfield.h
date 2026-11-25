/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   This software is distributed under the GNU General Public License.

   fix becfield: field-induced forces F_i[b] = sum_a eps_a Z*_i[a,b] from an EGCNN Born-effective-charge
   network, plus the accumulated polarization change.  Only the KOKKOS/CUDA version (becfield/kk) is
   implemented; this base class exists so that the style is registered and gives a clear error otherwise.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(becfield,FixBECField);
// clang-format on
#else

#ifndef LMP_FIX_BECFIELD_H
#define LMP_FIX_BECFIELD_H

#include "fix.h"

namespace LAMMPS_NS {

class FixBECField : public Fix {
 public:
  FixBECField(class LAMMPS *, int, char **);
  int setmask() override;
};

}    // namespace LAMMPS_NS

#endif
#endif
