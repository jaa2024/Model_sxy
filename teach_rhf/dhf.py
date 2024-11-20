import numpy as np
import scipy
from pyscf import gto, scf
import ctypes
import numpy.typing as npt
import scipy.constants
import scipy.linalg
from build import build_lib

# Keep the original _cint and argtypes setup
_cint = build_lib()

LIGHT_SPEED = 137.03599967994

argtypes = [
    np.ctypeslib.ndpointer(dtype=np.complex128, ndim=2),
    (ctypes.c_int * 2),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=2),
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=2),
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.double, ndim=1),
]


argtypes_2e = [
    np.ctypeslib.ndpointer(dtype=np.complex128, ndim=4),
    (ctypes.c_int * 4),
    np.ctypeslib.ndpointer(dtype=np.intc, ndim=2),
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.intc, ndim=2),
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.double, ndim=1),
    ctypes.c_void_p(0),
]

_cint.CINTcgto_spinor.restype = ctypes.c_int
_cint.CINTcgto_spinor.argtypes = [
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=2),
]

_cint.CINTtot_cgto_spinor.restype = ctypes.c_int
_cint.CINTtot_cgto_spinor.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=2),
    ctypes.c_int,
]


class DHF:
    def __init__(self, mol: gto.Mole):
        """Initialize RHF calculator with a PySCF Mole object."""
        self.mol = mol
        # Basic parameters
        self.atm = mol._atm.astype(np.intc)
        self.bas = mol._bas.astype(np.intc)
        self.env = mol._env.astype(np.double)
        self.nao = mol.nao_nr()
        self.n2c = mol.nao_2c()
        self.n4c = self.n2c * 2
        self.nshls = len(self.bas)
        self.natm = len(self.atm)
        self.nelec = sum(mol.nelec)
        self.ndocc = min(mol.nelec)

        # Initialize integral matrices
        self.S = None  # Overlap matrix
        self.H = None  # Core Hamiltonian
        # Dirac Coulumb integral
        self._coulomb_level = "LLLL"
        self.LLLL = None
        self.SSLL = None
        self.SSSS = None
        # Gaunt Breit integral
        self.with_gaunt = False
        self.with_briet = False  # not support!
        self.GAUNT = None
        self.BREIT = None  # not support!

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

        c = LIGHT_SPEED
        n2c = self.n2c
        n4c = self.n4c

        # Setup integral function arguments
        _cint.cint1e_spsp.argtypes = argtypes
        _cint.cint1e_nuc.argtypes = argtypes
        _cint.cint1e_spnucsp.argtypes = argtypes
        _cint.cint1e_ovlp.argtypes = argtypes

        # Initialize matrices
        t = np.zeros((self.n2c, self.n2c), np.complex128, order="F")
        vn = np.zeros((self.n2c, self.n2c), np.complex128, order="F")
        wn = np.zeros((self.n2c, self.n2c), np.complex128, order="F")
        s = np.zeros((self.n2c, self.n2c), np.complex128, order="F")
        # Compute one-electron integrals
        for i in range(self.nshls):
            di = _cint.CINTcgto_spinor(i, self.bas)
            x = _cint.CINTtot_cgto_spinor(self.bas, i)
            for j in range(i, self.nshls):
                dj = _cint.CINTcgto_spinor(j, self.bas)
                y = _cint.CINTtot_cgto_spinor(self.bas, j)

                # Allocate buffers
                buf_t = np.zeros((di, dj), np.complex128, order="F")
                buf_vn = np.zeros((di, dj), np.complex128, order="F")
                buf_wn = np.zeros((di, dj), np.complex128, order="F")
                buf_s = np.zeros((di, dj), np.complex128, order="F")

                # Compute integrals
                _cint.cint1e_spsp(
                    buf_t,
                    (ctypes.c_int * 2)(i, j),
                    self.atm,
                    self.natm,
                    self.bas,
                    self.nshls,
                    self.env,
                )
                _cint.cint1e_nuc(
                    buf_vn,
                    (ctypes.c_int * 2)(i, j),
                    self.atm,
                    self.natm,
                    self.bas,
                    self.nshls,
                    self.env,
                )
                _cint.cint1e_spnucsp(
                    buf_wn,
                    (ctypes.c_int * 2)(i, j),
                    self.atm,
                    self.natm,
                    self.bas,
                    self.nshls,
                    self.env,
                )
                _cint.cint1e_ovlp(
                    buf_s,
                    (ctypes.c_int * 2)(i, j),
                    self.atm,
                    self.natm,
                    self.bas,
                    self.nshls,
                    self.env,
                )
                # Store results
                t[x : x + di, y : y + dj] = buf_t
                vn[x : x + di, y : y + dj] = buf_vn
                wn[x : x + di, y : y + dj] = buf_wn
                s[x : x + di, y : y + dj] = buf_s

                t[y : y + dj, x : x + di] = buf_t.conj().T
                vn[y : y + dj, x : x + di] = buf_vn.conj().T
                wn[y : y + dj, x : x + di] = buf_wn.conj().T
                s[y : y + dj, x : x + di] = buf_s.conj().T

        self.H = np.zeros((n4c, n4c), np.complex128)
        self.H[:n2c, :n2c] = vn
        self.H[n2c:, :n2c] = t * 0.5
        self.H[:n2c, n2c:] = t * 0.5
        self.H[n2c:, n2c:] = wn * (0.25 / c**2) - t * 0.5

        self.S = np.zeros((n4c, n4c), np.complex128)
        self.S[:n2c, :n2c] = s
        self.S[n2c:, n2c:] = t * (0.25 / c**2)

        # Compute two-electron integrals
        print("Computing ERI integrals...")
        _cint.cint2e.argtypes = argtypes_2e
        _cint.cint2e_spsp1.argtypes = argtypes_2e
        _cint.cint2e_spsp1spsp2.argtypes = argtypes_2e

        self.LLLL = np.zeros((n2c, n2c, n2c, n2c), np.complex128)
        self.SSLL = np.zeros((n2c, n2c, n2c, n2c), np.complex128)
        self.SSSS = np.zeros((n2c, n2c, n2c, n2c), np.complex128)

        for i in range(self.nshls):
            di = _cint.CINTcgto_spinor(i, self.bas)
            x = _cint.CINTtot_cgto_spinor(self.bas, i)
            for j in range(self.nshls):
                dj = _cint.CINTcgto_spinor(j, self.bas)
                y = _cint.CINTtot_cgto_spinor(self.bas, j)
                for k in range(self.nshls):
                    dk = _cint.CINTcgto_spinor(k, self.bas)
                    z = _cint.CINTtot_cgto_spinor(self.bas, k)
                    for l in range(self.nshls):  # noqa: E741
                        dl = _cint.CINTcgto_spinor(l, self.bas)
                        w = _cint.CINTtot_cgto_spinor(self.bas, l)

                        llll = np.zeros((di, dj, dk, dl), np.complex128, order="F")
                        ssll = np.zeros((di, dj, dk, dl), np.complex128, order="F")
                        ssss = np.zeros((di, dj, dk, dl), np.complex128, order="F")
                        _cint.cint2e(
                            llll,
                            (ctypes.c_int * 4)(i, j, k, l),
                            self.atm,
                            self.natm,
                            self.bas,
                            self.nshls,
                            self.env,
                            ctypes.c_void_p(0),
                        )
                        _cint.cint2e_spsp1(
                            ssll,
                            (ctypes.c_int * 4)(i, j, k, l),
                            self.atm,
                            self.natm,
                            self.bas,
                            self.nshls,
                            self.env,
                            ctypes.c_void_p(0),
                        )
                        _cint.cint2e_spsp1spsp2(
                            ssss,
                            (ctypes.c_int * 4)(i, j, k, l),
                            self.atm,
                            self.natm,
                            self.bas,
                            self.nshls,
                            self.env,
                            ctypes.c_void_p(0),
                        )
                        self.LLLL[x : x + di, y : y + dj, z : z + dk, w : w + dl] = llll
                        self.SSLL[x : x + di, y : y + dj, z : z + dk, w : w + dl] = ssll
                        self.SSSS[x : x + di, y : y + dj, z : z + dk, w : w + dl] = ssss
        # Compute core Hamiltonian and orthogonalization matrix
        print("Integral computation completed.")

    def get_fock(self, D: npt.NDArray) -> npt.NDArray:
        """Build Fock matrix from density matrix using precomputed integrals."""

        n2c = self.n2c
        coulomb_level = self._coulomb_level
        c1 = 0.5 / LIGHT_SPEED

        vj = np.zeros((n2c * 2, n2c * 2), dtype=np.complex128)
        vk = np.zeros((n2c * 2, n2c * 2), dtype=np.complex128)
        if coulomb_level.upper() == "LLLL":
            dms = D[:n2c, :n2c].copy()
            J = np.einsum("ijkl,kl->ij", self.LLLL, dms, optimize=True)
            K = np.einsum("ikjl,kl->ij", self.LLLL, dms, optimize=True)

            vj[:n2c, :n2c] = J
            vk[:n2c, :n2c] = K
        elif coulomb_level.upper() == "SSLL" or coulomb_level.upper() == "LLSS":
            dm = D.copy()
            dm1 = dm[:n2c, :n2c].copy()
            dm2 = dm[n2c:, n2c:].copy()
            dm3 = dm[n2c:, :n2c].copy()
            dm4 = dm[:n2c, n2c:].copy()

            J1 = (
                np.einsum("ijkl,lk->ij", self.SSLL, dm1, optimize=True) * c1**2
            )  # lk->s2ij
            J2 = (
                np.einsum("ijkl,ji->kl", self.SSLL, dm2, optimize=True) * c1**2
            )  # ji->s2kl
            K1 = (
                np.einsum("ikjl,jk->il", self.SSLL, dm3, optimize=True) * c1**2
            )  # jk->s1il
            K2 = (
                np.einsum("ljik,li->kj", self.SSLL, dm4, optimize=True) * c1**2
            )  # li->s1kj

            dms = D[:n2c, :n2c].copy()
            J = np.einsum("ijkl,kl->ij", self.LLLL, dms, optimize=True)
            K = np.einsum("ikjl,kl->ij", self.LLLL, dms, optimize=True)

            vj[n2c:, n2c:] = J1
            vj[:n2c, :n2c] = J2
            vk[n2c:, :n2c] = K1
            vk[:n2c, n2c:] = K2

            vj[:n2c, :n2c] += J
            vk[:n2c, :n2c] += K
        else:
            dm = D.copy()
            dm1 = dm[:n2c, :n2c].copy()
            dm2 = dm[n2c:, n2c:].copy()
            dm3 = dm[n2c:, :n2c].copy()
            dm4 = dm[:n2c, n2c:].copy()

            J1 = (
                np.einsum("ijkl,lk->ij", self.SSLL, dm1, optimize=True) * c1**2
            )  # lk->s2ij
            J2 = (
                np.einsum("ijkl,ji->kl", self.SSLL, dm2, optimize=True) * c1**2
            )  # ji->s2kl
            K1 = (
                np.einsum("ikjl,jk->il", self.SSLL, dm3, optimize=True) * c1**2
            )  # jk->s1il
            K2 = (
                np.einsum("ljik,li->kj", self.SSLL, dm4, optimize=True) * c1**2
            )  # li->s1kj

            dms = D[:n2c, :n2c].copy()
            J = np.einsum("ijkl,kl->ij", self.LLLL, dms, optimize=True)
            K = np.einsum("ikjl,kl->ij", self.LLLL, dms, optimize=True)

            vj[n2c:, n2c:] = J1
            vj[:n2c, :n2c] = J2
            vk[n2c:, :n2c] = K1
            vk[:n2c, n2c:] = K2

            vj[:n2c, :n2c] += J
            vk[:n2c, :n2c] += K

            dms = D[n2c:, n2c:].copy()
            J = np.einsum("ijkl,kl->ij", self.SSSS, dms, optimize=True) * c1**4
            K = np.einsum("ikjl,kl->ij", self.LLLL, dms, optimize=True) * c1**4

            vj[n2c:, n2c:] += J
            vk[n2c:, n2c:] += K

        return self.H + vj - vk

    def make_density(self, fock: npt.NDArray) -> npt.NDArray:
        """Create new density matrix and calculate electronic energy."""
        # Solve eigenvalue problem
        mo_energy, mo_coeff = scipy.linalg.eigh(fock, self.S)

        n4c = self.n4c
        n2c = self.n2c

        mo_occ = np.zeros(n4c)
        c = LIGHT_SPEED
        collapse_threshold = -1.999 * c**2

        if mo_energy[n2c] > collapse_threshold:
            # Normal case - fill lowest energy orbitals
            mo_occ[n2c : n2c + self.nelec] = 1
        else:
            # Handle variational collapse
            valid_energies = mo_energy > collapse_threshold
            lumo = mo_energy[valid_energies][self.nelec]
            mo_occ[valid_energies] = 1
            mo_occ[mo_energy >= lumo] = 0

        # Form density matrix
        C_occ = mo_coeff[:, mo_occ > 0]
        # occ_weights = mo_occ[mo_occ > 0]

        # return np.einsum("pi,i,qi->pq", C_occ, occ_weights, C_occ, optimize=True)
        return (C_occ * mo_occ[mo_occ > 0]).dot(C_occ.conj().T)

    def get_energy_elec(self, F: npt.NDArray, D: npt.NDArray) -> float:
        return np.einsum("pq,pq->", (self.H + F), D, optimize=True)

    def get_energy_tot(self, F: npt.NDArray, D: npt.NDArray) -> float:
        return self.get_energy_elec(F, D) + self.E_nn

    def scf(
        self, max_iter: int = 100, conv_tol: float = 1e-6, D: npt.NDArray = None
    ) -> npt.NDArray:
        if D is None:
            D = self.build_init_guess()

        for iter_num in range(max_iter):
            F = self.get_fock(D)
        return D

    def kernel(self, max_iter: int = 100, conv_tol: float = 1e-6) -> float:
        """Run the SCF procedure with precomputed integrals."""
        # Precompute all integrals before starting SCF
        self._compute_all_integrals()

        print("Starting SCF calculation...")
        # Initial guess using core Hamiltonian
        D = self.make_density(self.H)
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

        raise RuntimeError("SCF did not converge within maximum iterations")


def main():
    # Example usage
    mol = gto.M(atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587")
    mol.basis = {
        "H": [
            [0, 0, [8.29687389e01, 1.0]],
            [0, 0, [1.24571508e01, 1.0]],
            [0, 0, [2.83382422e00, 1.0]],
            [0, 0, [8.00164139e-01, 1.0]],
            [0, 0, [2.58629441e-01, 1.0]],
            [0, 0, [8.99766770e-02, 1.0]],
            [1, 0, [5.02448897e-01, 1.0]],
        ],
        "O": [
            [0, 0, [1.59925081e04, 1.0]],
            [0, 0, [2.35199842e03, 1.0]],
            [0, 0, [5.29060079e02, 1.0]],
            [0, 0, [1.48462548e02, 1.0]],
            [0, 0, [4.78417313e01, 1.0]],
            [0, 0, [1.68559662e01, 1.0]],
            [0, 0, [6.23711537e00, 1.0]],
            [0, 0, [1.75808802e00, 1.0]],
            [0, 0, [6.90688830e-01, 1.0]],
            [0, 0, [2.39078229e-01, 1.0]],
            [1, 0, [6.33409745e01, 1.0]],
            [1, 0, [1.45830479e01, 1.0]],
            [1, 0, [4.42797374e00, 1.0]],
            [1, 0, [1.51782600e00, 1.0]],
            [1, 0, [5.24175730e-01, 1.0]],
            [1, 0, [1.72133105e-01, 1.0]],
            [2, 0, [1.17766132e00, 1.0]],
        ],
    }

    mf = DHF(mol)
    mf._compute_all_integrals()

    # ref = mol.intor("int2e_ssp1ssp2_spinor").reshape(mf.n2c, mf.n2c, mf.n2c, mf.n2c)

    # print(np.sum(ref - mf.SSSS))
    # Compare with PySCF
    mf_pyscf = scf.DHF(mol)
    mf_pyscf.verbose = 0
    mf_pyscf.init_guess = "1e"

    # print(mol.time_reversal_map())
    # print("E(Dirac-Coulomb) = %.15g" % mf_pyscf.kernel())

    # mf_pyscf.with_gaunt = True
    # print("E(Dirac-Coulomb-Gaunt) = %.15g" % mf_pyscf.kernel())

    #
    # mf_pyscf.with_breit = True
    # print("E(Dirac-Coulomb-Breit) = %.15g" % mf_pyscf.kernel())

    #
    # # Our implementation
    # mf = RHF(mol)
    # E_our = mf.kernel()
    # print(f"Our energy:   {E_our:.10f}")
    # print(f"Difference:   {abs(E_pyscf - E_our):.10f}")


if __name__ == "__main__":
    main()
