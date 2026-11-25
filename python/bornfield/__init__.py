"""BM-BornField (Python): EGCNN Born effective charges on the GPU, with an ASE calculator.

Same CUDA network as the LAMMPS fix becfield/kk (lammps/becfield_core.cuh), with its own GPU cell-list neighbour list.

    from bornfield import BornField, PolarizationTracker
    bf = BornField("egcnn_bto.json", electric_field=(0, 0, 0.03))       # V/Angstrom
    atoms.calc = SumCalculator([mace_calc, bf])                          # zero-field MLP + field force
    Z = bf.get_property("becs", atoms)                                   # (N, 3, 3), Z[i, a, b]

Conventions (identical to efmlp_lammps.BECFieldForce and fix becfield/kk):
  Z[i, a, b]: a = field / polarization direction, b = displacement direction; acoustic sum rule applied.
  forces F_i[b] = sum_a eps_a Z_i[a, b]; no energy and no stress (reported as 0).
  PolarizationTracker: dP_a += sum_i 1/2 (Z_i(t) + Z_i(t-dt))[a, b] dr_i[b] (minimum image, mean cell), e*Angstrom.
"""
import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from ._bornfield import BornFieldGPU

__all__ = ["BornFieldGPU", "BornField", "PolarizationTracker"]


class BornField(Calculator):
    """ASE calculator: field force eps . Z* from an exported EGCNN model (tools/export_egcnn.py).

    electric_field: (3,) V/Angstrom, or a callable f(atoms) -> (3,); atoms.info["electric_field"] overrides it.
    precision: "float32" (default, as in training) or "float64"; radial: "spline" (default) or "exact".
    energy/free_energy/stress are returned as 0 (the BEC x field force is not conservative)."""

    implemented_properties = ["energy", "free_energy", "forces", "stress", "becs"]

    def __init__(self, model, electric_field=(0.0, 0.0, 0.0), precision="float32", radial="spline", grid=1024,
                 **kwargs):
        super().__init__(**kwargs)
        self.model = BornFieldGPU(str(model), precision, radial, int(grid))
        self.electric_field = electric_field

    def field(self, atoms):
        e = atoms.info.get("electric_field", self.electric_field)
        if callable(e):
            e = e(atoms)
        return np.asarray(e, dtype=float).reshape(3)

    def becs(self, atoms):
        Z = self.model.compute(np.ascontiguousarray(atoms.positions, dtype=float),
                               np.ascontiguousarray(atoms.numbers, dtype=np.int32),
                               np.ascontiguousarray(atoms.cell.array, dtype=float),
                               [bool(p) for p in atoms.pbc])
        return Z.reshape(-1, 3, 3)

    def calculate(self, atoms=None, properties=("forces",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        Z = self.becs(self.atoms)
        eps = self.field(self.atoms)
        self.results = {"becs": Z, "forces": np.einsum("a,iab->ib", eps, Z), "energy": 0.0, "free_energy": 0.0,
                        "stress": np.zeros(6), "electric_field": eps}


class PolarizationTracker:
    """Accumulated polarization change (e*Angstrom) along a trajectory, as efmlp_lammps.BECFieldForce / fix becfield.

    Call update(atoms, Z) after every force evaluation (e.g. dyn.attach(lambda: tr.update(atoms), interval=1)
    with a BornField calculator in atoms.calc: Z is then taken from its last results)."""

    def __init__(self, P0=(0.0, 0.0, 0.0)):
        self.P = np.asarray(P0, dtype=float).copy()
        self.prev = None

    def update(self, atoms, Z=None):
        if Z is None:
            Z = self._find_becs(atoms.calc)
        cell = atoms.cell.array.copy()
        frac = np.linalg.solve(cell.T, atoms.positions.T).T
        if self.prev is not None:
            f0, Z0, c0 = self.prev
            ds = frac - f0
            ds -= np.round(ds)
            self.P += np.einsum("iab,ib->a", 0.5 * (Z + Z0), ds @ (0.5 * (cell + c0)))
        self.prev = (frac, np.array(Z, copy=True), cell)
        return self.P.copy()

    @staticmethod
    def _find_becs(calc):
        if "becs" in getattr(calc, "results", {}):
            return calc.results["becs"]
        calcs = getattr(calc, "calcs", None) or getattr(getattr(calc, "mixer", None), "calcs", None) or []
        for c in calcs:
            if "becs" in getattr(c, "results", {}):
                return c.results["becs"]
        raise RuntimeError("no BornField results found on atoms.calc; pass Z explicitly")
