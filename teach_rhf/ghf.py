import numpy as np
import scipy.linalg
from pyscf import gto, scf
import ctypes
import numpy.typing as npt
from build import build_lib

# Keep the original _cint and argtypes setup
_cint = build_lib()

argtypes = [
    np.ctypeslib.ndpointer(dtype=np.double, ndim=2),
    (ctypes.c_int * 2),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=2),
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=2),
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.double, ndim=1),
]

_cint.CINTcgto_spheric.restype = ctypes.c_int
_cint.CINTcgto_spheric.argtypes = [
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=2),
]

_cint.CINTtot_cgto_spheric.restype = ctypes.c_int
_cint.CINTtot_cgto_spheric.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=2),
    ctypes.c_int,
]


class GHF:
    def __init__(self, mol: gto.Mole):
        """Initialize RHF calculator with a PySCF Mole object."""
        self.mol = mol
        # Basic parameters
        self.atm = mol._atm.astype(np.intc)
        self.bas = mol._bas.astype(np.intc)
        self.env = mol._env.astype(np.double)
        self.nao = mol.nao_nr()
        self.nso = 2 * self.nao
        self.nshls = len(self.bas)
        self.natm = len(self.atm)
        self.nalpha = mol.nelec[0]
        self.nbeta = mol.nelec[1]
        self.nelec = self.nalpha + self.nbeta
        self.ndocc = min(self.nalpha, self.nbeta)
        self.nsocc = abs(self.nalpha - self.nbeta)

        # Initialize integral matrices
        self.S = None  # Overlap matrix
        self.T = None  # Kinetic matrix
        self.V = None  # Nuclear attraction matrix
        self.H = None  # Core Hamiltonian
        self.eri = None  # Electron repulsion integrals

        # Nuclear repulsion energy
        self.E_nn = self._compute_nuclear_repulsion()

    def _compute_nuclear_repulsion(self) -> float:
        """Calculate nuclear repulsion energy."""
        coords = self.mol.atom_coords()
        charges = self.mol.atom_charges()
        natm = len(charges)
        E_nn = 0.0
        for i in range(natm):
            for j in range(i + 1, natm):
                r_ij = np.linalg.norm(coords[i] - coords[j])
                E_nn += charges[i] * charges[j] / r_ij
        return E_nn

    def _compute_all_integrals(self):
        """Precompute all necessary integrals."""
        print("Precomputing integrals...")

        # Setup integral function arguments
        _cint.cint1e_ovlp_sph.argtypes = argtypes
        _cint.cint1e_kin_sph.argtypes = argtypes
        _cint.cint1e_nuc_sph.argtypes = argtypes

        # Initialize matrices
        self.S = np.zeros((self.nao, self.nao))
        self.T = np.zeros((self.nao, self.nao))
        self.V = np.zeros((self.nao, self.nao))
        self.eri = np.zeros((self.nao,) * 4)

        # Compute one-electron integrals
        for i in range(self.nshls):
            di = _cint.CINTcgto_spheric(i, self.bas)
            x = _cint.CINTtot_cgto_spheric(self.bas, i)
            for j in range(i, self.nshls):
                dj = _cint.CINTcgto_spheric(j, self.bas)
                y = _cint.CINTtot_cgto_spheric(self.bas, j)

                # Allocate buffers
                buf_s = np.empty((di, dj), order="F")
                buf_t = np.empty((di, dj), order="F")
                buf_v = np.empty((di, dj), order="F")

                # Compute integrals
                _cint.cint1e_ovlp_sph(
                    buf_s,
                    (ctypes.c_int * 2)(i, j),
                    self.atm,
                    self.natm,
                    self.bas,
                    self.nshls,
                    self.env,
                )
                _cint.cint1e_kin_sph(
                    buf_t,
                    (ctypes.c_int * 2)(i, j),
                    self.atm,
                    self.natm,
                    self.bas,
                    self.nshls,
                    self.env,
                )
                _cint.cint1e_nuc_sph(
                    buf_v,
                    (ctypes.c_int * 2)(i, j),
                    self.atm,
                    self.natm,
                    self.bas,
                    self.nshls,
                    self.env,
                )

                # Store results
                self.S[x : x + di, y : y + dj] = buf_s
                self.T[x : x + di, y : y + dj] = buf_t
                self.V[x : x + di, y : y + dj] = buf_v

                self.S[y : y + dj, x : x + di] = buf_s.T
                self.T[y : y + dj, x : x + di] = buf_t.T
                self.V[y : y + dj, x : x + di] = buf_v.T

        # Compute two-electron integrals
        print("Computing ERI integrals...")
        CINTOpt = ctypes.c_void_p
        opt = CINTOpt()

        _cint.cint2e_sph_optimizer.argtypes = [
            ctypes.POINTER(CINTOpt),
            np.ctypeslib.ndpointer(dtype=np.intc, ndim=2),
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.intc, ndim=2),
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.double, ndim=1),
        ]

        _cint.cint2e_sph_optimizer(
            ctypes.byref(opt), self.atm, self.natm, self.bas, self.nshls, self.env
        )
        _cint.cint2e_sph.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double, ndim=4),
            (ctypes.c_int * 4),
            np.ctypeslib.ndpointer(dtype=np.intc, ndim=2),
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.intc, ndim=2),
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.double, ndim=1),
            CINTOpt,
        ]
        for i in range(self.nshls):
            di = _cint.CINTcgto_spheric(i, self.bas)
            x = _cint.CINTtot_cgto_spheric(self.bas, i)
            for j in range(i, self.nshls):
                dj = _cint.CINTcgto_spheric(j, self.bas)
                y = _cint.CINTtot_cgto_spheric(self.bas, j)
                for k in range(self.nshls):
                    dk = _cint.CINTcgto_spheric(k, self.bas)
                    z = _cint.CINTtot_cgto_spheric(self.bas, k)
                    for l in range(k, self.nshls):  # noqa: E741
                        dl = _cint.CINTcgto_spheric(l, self.bas)
                        w = _cint.CINTtot_cgto_spheric(self.bas, l)

                        buf = np.empty((di, dj, dk, dl), order="F")
                        _cint.cint2e_sph(
                            buf,
                            (ctypes.c_int * 4)(i, j, k, l),
                            self.atm,
                            self.natm,
                            self.bas,
                            self.nshls,
                            self.env,
                            opt,
                        )
                        self.eri[x : x + di, y : y + dj, z : z + dk, w : w + dl] = (
                            buf.transpose(0, 1, 2, 3)
                        )
                        self.eri[y : y + dj, x : x + di, z : z + dk, w : w + dl] = (
                            buf.transpose(1, 0, 2, 3)
                        )
                        self.eri[x : x + di, y : y + dj, w : w + dl, z : z + dk] = (
                            buf.transpose(0, 1, 3, 2)
                        )
                        self.eri[y : y + dj, x : x + di, w : w + dl, z : z + dk] = (
                            buf.transpose(1, 0, 3, 2)
                        )
                        self.eri[z : z + dk, w : w + dl, x : x + di, y : y + dj] = (
                            buf.transpose(2, 3, 0, 1)
                        )
                        self.eri[w : w + dl, z : z + dk, x : x + di, y : y + dj] = (
                            buf.transpose(3, 2, 0, 1)
                        )
                        self.eri[z : z + dk, w : w + dl, y : y + dj, x : x + di] = (
                            buf.transpose(2, 3, 1, 0)
                        )
                        self.eri[w : w + dl, z : z + dk, y : y + dj, x : x + di] = (
                            buf.transpose(3, 2, 1, 0)
                        )

        # Compute core Hamiltonian and orthogonalization matrix
        self.S = scipy.linalg.block_diag(self.S, self.S)
        self.H = self.T + self.V
        self.H = scipy.linalg.block_diag(self.H, self.H)
        print("Integral computation completed.")

    def build_init_guess(self) -> npt.NDArray:
        _, C = scipy.linalg.eigh(self.H, self.S)
        C_occ = C[:, : self.nelec]
        return np.einsum("pi,qi->pq", C_occ, C_occ, optimize=True) + 0.05

    def make_density(self, fock: npt.NDArray) -> npt.NDArray:
        # Solve eigenvalue problem
        _, C = scipy.linalg.eigh(fock, self.S)
        C_occ = C[:, : self.nelec]
        return np.einsum("pi,qi->pq", C_occ, C_occ, optimize=True)

    def get_fock(self, D: npt.NDArray) -> npt.NDArray:
        nao = self.nao
        dmaa = D[:nao, :nao]
        dmab = D[:nao, nao:]
        dmbb = D[nao:, nao:]

        J = np.einsum("rs, pqrs -> pq", dmaa, self.eri, optimize=True) + np.einsum(
            "rs, pqrs -> pq", dmbb, self.eri, optimize=True
        )
        Kaa = np.einsum("rs, prqs -> pq", dmaa, self.eri, optimize=True)
        Kbb = np.einsum("rs, prqs -> pq", dmbb, self.eri, optimize=True)
        Kab = np.einsum("rs, prqs -> pq", dmab, self.eri, optimize=True)
        return np.block([[J - Kaa, -Kab], [-Kab, J - Kbb]]) + self.H

    def get_energy_elec(self, F: npt.NDArray, D: npt.NDArray) -> float:
        return np.einsum("pq,pq->", (self.H + F), D, optimize=True) / 2

    def get_energy_tot(self, F: npt.NDArray, D: npt.NDArray) -> float:
        return self.get_energy_elec(F, D) + self.E_nn

    def kernel(self, max_iter: int = 1000, conv_tol: float = 1e-6):
        """Run the SCF procedure with precomputed integrals."""
        # Precompute all integrals before starting SCF
        self._compute_all_integrals()

        print("Starting SCF calculation...")
        # Initial guess using core Hamiltonian
        D = self.build_init_guess()
        E_old = 0.0

        for iter_num in range(max_iter):
            # Build Fock matrix
            F = self.get_fock(D)
            E_total = self.get_energy_tot(F, D)
            # Get new density matrix and energy
            D_new = self.make_density(F)

            # Check convergence
            E_diff = abs(E_total - E_old)
            D_diff = np.linalg.norm(D_new - D)

            print(
                f"Iter {iter_num:3d}: E = {E_total:.10f}, "
                f"dE = {E_diff:.3e}, dD = {D_diff:.3e}"
            )

            if E_diff < conv_tol and D_diff < conv_tol:
                print("\nSCF Converged!")
                print(f"Final SCF energy: {E_total:.10f}")
                return E_total

            D = D_new
            E_old = E_total

        print("\nSCF Converged!")
        print(f"Final SCF energy: {E_old:.10f}")
        raise RuntimeError("SCF did not converge within maximum iterations")


def main():
    # Example usage
    mol = gto.M(atom="O 0 0 0; O 0 0 1.2", basis="ccpvdz", spin=2)

    # Compare with PySCF
    mf_pyscf = scf.GHF(mol)
    E_pyscf = mf_pyscf.kernel()
    print(f"\nPySCF energy: {E_pyscf:.10f}")

    # Our implementation
    mf = GHF(mol)
    E_our = mf.kernel()
    print(f"Our energy:   {E_our:.10f}")
    print(f"Difference:   {abs(E_pyscf - E_our):.10f}")


if __name__ == "__main__":
    main()
