/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   This software is distributed under the GNU General Public License.
------------------------------------------------------------------------- */

#include "fix_becfield.h"

#include "error.h"

using namespace LAMMPS_NS;

FixBECField::FixBECField(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{
  if (!kokkosable)
    error->all(FLERR, "fix becfield is only implemented for KOKKOS on a CUDA GPU: "
                      "run with '-k on g 1 -sf kk' (or use fix becfield/kk)");
}

int FixBECField::setmask()
{
  return 0;
}
