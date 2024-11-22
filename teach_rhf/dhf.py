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
        self.with_ssss = True
        self.SSSS = None
        # Gaunt Breit integral

        self.with_gaunt = True  # TODO
        self.with_briet = False  # not support!
        self.LSLS = None
        self.SLSL = None
        self.LSSL = None
        self.SLLS = None

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
        _cint.cint2e.argtypes = argtypes_2e  # (LL|LL)
        _cint.cint2e_spsp1.argtypes = argtypes_2e  # (SS|LL)
        # (SS|SS)
        _cint.cint2e_spsp1spsp2.argtypes = argtypes_2e  # (SS|SS)
        # Gaunt
        _cint.cint2e_ssp1ssp2.argtypes = argtypes_2e  # (LσS|LσS)
        # _cint.cint2e_sps1sps2.argtypes = argtypes_2e  # (SσL|SσL)
        _cint.cint2e_ssp1sps2.argtypes = argtypes_2e  # (LσS|SσL)
        # _cint.cint2e_sps1ssp2.argtypes = argtypes_2e  # (SσL|LσS)

        self.LLLL = np.zeros((n2c, n2c, n2c, n2c), np.complex128)
        self.SSLL = np.zeros((n2c, n2c, n2c, n2c), np.complex128)

        if self.with_ssss is True:
            self.SSSS = np.zeros((n2c, n2c, n2c, n2c), np.complex128)
        if self.with_gaunt is True:
            self.LSLS = np.zeros((n2c, n2c, n2c, n2c), np.complex128)
            self.LSSL = np.zeros((n2c, n2c, n2c, n2c), np.complex128)

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

                        self.LLLL[x : x + di, y : y + dj, z : z + dk, w : w + dl] = llll
                        self.SSLL[x : x + di, y : y + dj, z : z + dk, w : w + dl] = ssll

                        if self.with_ssss is True:
                            ssss = np.zeros((di, dj, dk, dl), np.complex128, order="F")

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
                            self.SSSS[
                                x : x + di, y : y + dj, z : z + dk, w : w + dl
                            ] = ssss
                        if self.with_gaunt is True:
                            lsls = np.zeros((di, dj, dk, dl), np.complex128, order="F")
                            # slsl = np.zeros((di, dj, dk, dl), np.complex128, order="F")
                            lssl = np.zeros((di, dj, dk, dl), np.complex128, order="F")
                            # slls = np.zeros((di, dj, dk, dl), np.complex128, order="F")

                            _cint.cint2e_ssp1ssp2(
                                lsls,
                                (ctypes.c_int * 4)(i, j, k, l),
                                self.atm,
                                self.natm,
                                self.bas,
                                self.nshls,
                                self.env,
                                ctypes.c_void_p(0),
                            )

                            # _cint.cint2e_sps1sps2(
                            #     slsl,
                            #     (ctypes.c_int * 4)(i, j, k, l),
                            #     self.atm,
                            #     self.natm,
                            #     self.bas,
                            #     self.nshls,
                            #     self.env,
                            #     ctypes.c_void_p(0),
                            # )

                            _cint.cint2e_ssp1sps2(
                                lssl,
                                (ctypes.c_int * 4)(i, j, k, l),
                                self.atm,
                                self.natm,
                                self.bas,
                                self.nshls,
                                self.env,
                                ctypes.c_void_p(0),
                            )

                            # _cint.cint2e_sps1ssp2(
                            #     slls,
                            #     (ctypes.c_int * 4)(i, j, k, l),
                            #     self.atm,
                            #     self.natm,
                            #     self.bas,
                            #     self.nshls,
                            #     self.env,
                            #     ctypes.c_void_p(0),
                            # )

                            self.LSLS[
                                x : x + di, y : y + dj, z : z + dk, w : w + dl
                            ] = lsls
                            # self.SLSL[
                            #     x : x + di, y : y + dj, z : z + dk, w : w + dl
                            # ] = slsl
                            self.LSSL[
                                x : x + di, y : y + dj, z : z + dk, w : w + dl
                            ] = lssl
                            # self.SLLS[
                            #     x : x + di, y : y + dj, z : z + dk, w : w + dl
                            # ] = slls

        # Compute core Hamiltonian and orthogonalization matrix
        print("Integral computation completed.")

    def build_init_guess(self):
        return self.make_density(self.H)

    def _call_veff_LLLL(self, D: npt.NDArray):
        n2c = self.n2c
        vj = np.zeros((n2c * 2, n2c * 2), dtype=np.complex128)
        vk = np.zeros((n2c * 2, n2c * 2), dtype=np.complex128)

        dms = D[:n2c, :n2c].copy()
        J = np.einsum("ijkl,lk->ij", self.LLLL, dms, optimize=True)
        K = np.einsum("ilkj,lk->ij", self.LLLL, dms, optimize=True)

        vj[:n2c, :n2c] = J
        vk[:n2c, :n2c] = K

        return vj, vk

    def _call_veff_SSLL(self, D: npt.NDArray):
        n2c = self.n2c
        c1 = 0.5 / LIGHT_SPEED
        vj = np.zeros((n2c * 2, n2c * 2), dtype=np.complex128)
        vk = np.zeros((n2c * 2, n2c * 2), dtype=np.complex128)

        dm = D.copy()
        dmll = dm[:n2c, :n2c].copy()
        dmss = dm[n2c:, n2c:].copy()
        dmsl = dm[n2c:, :n2c].copy()
        dmls = dm[:n2c, n2c:].copy()

        J1 = np.einsum("ijkl,lk->ij", self.SSLL, dmll, optimize=True) * c1**2
        J2 = np.einsum("klij,lk->ij", self.SSLL, dmss, optimize=True) * c1**2
        K1 = np.einsum("ilkj,lk->ij", self.SSLL, dmsl, optimize=True) * c1**2
        # K2 = np.einsum("kjil,lk->ij", self.SSLL, dmls, optimize=True) * c1**2
        K2 = K1.transpose().conj()

        vj[n2c:, n2c:] = J1
        vj[:n2c, :n2c] = J2
        vk[n2c:, :n2c] = K1
        vk[:n2c, n2c:] = K2

        return vj, vk

    def _call_veff_SSSS(self, D: npt.NDArray):
        n2c = self.n2c
        c1 = 0.5 / LIGHT_SPEED

        vj = np.zeros((n2c * 2, n2c * 2), dtype=np.complex128)
        vk = np.zeros((n2c * 2, n2c * 2), dtype=np.complex128)

        dms = D[n2c:, n2c:].copy()
        J = np.einsum("ijkl,lk->ij", self.SSSS, dms, optimize=True) * c1**4
        K = np.einsum("ilkj,lk->ij", self.SSSS, dms, optimize=True) * c1**4

        vj[n2c:, n2c:] += J
        vk[n2c:, n2c:] += K

        return vj, vk

    def _call_veff_gaunt(self, D: npt.NDArray):
        n2c = self.n2c
        c1 = 0.5 / LIGHT_SPEED
        vj = np.zeros((n2c * 2, n2c * 2), dtype=np.complex128)
        vk = np.zeros((n2c * 2, n2c * 2), dtype=np.complex128)

        dm = D.copy()
        dmll = dm[:n2c, :n2c].copy()
        dmss = dm[n2c:, n2c:].copy()
        dmsl = dm[n2c:, :n2c].copy()
        dmls = dm[:n2c, n2c:].copy()

        # Kss = np.einsum("ilkj,lk->ij", self.SLLS, dmll, optimize=True) * c1**2
        Kll = np.einsum("ilkj,lk->ij", self.LSSL, dmss, optimize=True) * c1**2
        Kss = np.einsum("kjil,lk->ij", self.LSSL, dmll, optimize=True) * c1**2
        Kls = np.einsum("ilkj,lk->ij", self.LSLS, dmsl, optimize=True) * c1**2
        # Ksl = np.einsum("ilkj,lk->ij", self.SLSL, dmls, optimize=True) * c1**2
        Ksl = Kls.transpose().conj()

        Jls = (
            np.einsum("ijkl,lk->ij", self.LSSL, dmls, optimize=True)
            + np.einsum("ijkl,lk->ij", self.LSLS, dmsl, optimize=True)
        ) * c1**2
        # Jsl = (
        #     np.einsum("ijkl,lk->ij", self.SLLS, dmsl, optimize=True)
        #     + np.einsum("ijkl,lk->ij", self.SLSL, dmls, optimize=True)
        # ) * c1**2
        Jsl = Jls.transpose().conj()

        vk[:n2c, :n2c] = Kll
        vk[n2c:, n2c:] = Kss
        vk[:n2c, n2c:] = Kls
        vk[n2c:, :n2c] = Ksl

        vj[:n2c, n2c:] = Jls
        vj[n2c:, :n2c] = Jsl

        return vj, vk

    def get_vhf(self, D: npt.NDArray) -> npt.NDArray:
        """Build Fock matrix from density matrix using precomputed integrals."""

        coulomb_level = self._coulomb_level

        if coulomb_level.upper() == "LLLL":
            vj, vk = self._call_veff_LLLL(D)

        elif coulomb_level.upper() == "SSLL" or coulomb_level.upper() == "LLSS":
            vj, vk = self._call_veff_SSLL(D)
            J, K = self._call_veff_LLLL(D)
            vj += J
            vk += K

            if self.with_gaunt:
                J, K = self._call_veff_gaunt(D)
                vj -= J
                vk -= K

        else:  # SSSS
            vj, vk = self._call_veff_SSLL(D)
            J, K = self._call_veff_LLLL(D)
            vj += J
            vk += K

            J, K = self._call_veff_SSSS(D)

            vj += J
            vk += K

            if self.with_gaunt:
                J, K = self._call_veff_gaunt(D)
                vj -= J
                vk -= K

        return vj - vk

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

        return (C_occ * mo_occ[mo_occ > 0]).dot(C_occ.conj().T)

    def get_energy_elec(self, V: npt.NDArray, D: npt.NDArray) -> float:
        e1 = np.einsum("pq,qp->", self.H, D, optimize=True).real
        e_col = np.einsum("pq,qp->", V, D, optimize=True).real * 0.5
        return e1 + e_col

    def get_energy_tot(self, V: npt.NDArray, D: npt.NDArray) -> float:
        return self.get_energy_elec(V, D) + self.E_nn

    def scf(
        self, max_iter: int = 100, conv_tol: float = 1e-6, D: npt.NDArray = None
    ) -> npt.NDArray:
        if D is None:
            D = self.build_init_guess()

        E_old = 0.0
        print(f"Starting {self._coulomb_level}")

        for iter_num in range(max_iter):
            vhf = self.get_vhf(D)
            E_total = self.get_energy_tot(vhf, D)

            D_new = self.make_density(self.H + vhf)

            E_diff = E_total - E_old
            D_diff = np.mean((D_new - D).real ** 2) ** 0.5

            print(
                f"Iter {iter_num:3d}: E = {E_total:.10f}, "
                f"dE = {E_diff:.3e}, dD = {D_diff:.3e}"
            )

            if abs(E_diff) < conv_tol and D_diff < conv_tol:
                print("\nSCF Converged!")
                print(f"Final SCF energy: {E_total:.10f}")
                return D_new, E_total
            D = D_new
            E_old = E_total

        raise RuntimeError("SCF did not converge within maximum iterations")

    def kernel(self, max_iter: int = 100, conv_tol: float = 1e-6) -> float:
        """Run the SCF procedure with precomputed integrals."""
        # Precompute all integrals before starting SCF
        self._compute_all_integrals()
        self._coulomb_level = "LLLL"
        dm, e = self.scf(max_iter=100, conv_tol=1e-3, D=None)
        if self.with_ssss is True:
            self._coulomb_level = "SSLL"
            dm, e = self.scf(max_iter=100, conv_tol=1e-4, D=dm)

            self._coulomb_level = "SSSS"
            dm, e = self.scf(max_iter=100, conv_tol=conv_tol, D=dm)
        else:
            self._coulomb_level = "SSLL"
            dm, e = self.scf(max_iter=100, conv_tol=conv_tol, D=dm)
        return e


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

    E_our = mf.kernel()

    # ref = mol.intor("int2e_ssp1sps2_spinor").reshape(mf.n2c, mf.n2c, mf.n2c, mf.n2c)
    # print(ref.shape)
    # print(np.abs(ref - mf.LSSL).max())
    # Compare with PySCF
    mf_pyscf = scf.DHF(mol)

    # mf_pyscf.verbose = 5
    mf_pyscf.init_guess = "1e"
    mf_pyscf.with_ssss = True
    mf_pyscf.with_gaunt = True
    E_pyscf = mf_pyscf.kernel()

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
    print(f"\n\nOur energy:   {E_our:.10f}")
    print(f"PySCF energy: {E_pyscf:.10f}")
    print(f"Difference:   {abs(E_pyscf - E_our):.10f}")


if __name__ == "__main__":
    main()
