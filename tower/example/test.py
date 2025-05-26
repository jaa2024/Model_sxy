import Tower
from pyscf import gto, scf
import numpy as np

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
mol.basis = "sto-3g"
mol.build()

mf = scf.RHF(mol)
mf.kernel()


def get_bp_hso2e(mol, dm0):
    hso2e = mol.intor("int2e_p1vxp1", 3).reshape(3, mol.nao, mol.nao, mol.nao, mol.nao)
    vj = np.einsum("yijkl,lk->yij", hso2e, dm0, optimize=True)
    vk = np.einsum("yijkl,jk->yil", hso2e, dm0, optimize=True)
    vk += np.einsum("yijkl,li->ykj", hso2e, dm0, optimize=True)
    return vj, vk


def get_bp_hso2e_amfi(mol, dm0):
    """atomic-mean-field approximation"""
    ao_loc = mol.ao_loc_nr()
    nao = ao_loc[-1]
    vj = np.zeros((3, nao, nao))
    vk = np.zeros((3, nao, nao))
    import copy

    atom = copy.copy(mol)
    aoslice = mol.aoslice_by_atom(ao_loc)
    for ia in range(mol.natm):
        b0, b1, p0, p1 = aoslice[ia]
        atom._bas = mol._bas[b0:b1]
        vj1, vk1 = get_bp_hso2e(atom, dm0[p0:p1, p0:p1])
        vj[:, p0:p1, p0:p1] = vj1
        vk[:, p0:p1, p0:p1] = vk1
    return vj, vk


def get_bp_hso2e_amfi_cpp(mol, dm0):
    """atomic-mean-field approximation"""
    ao_loc = mol.ao_loc_nr()
    nao = ao_loc[-1]
    vj = np.zeros((3, nao, nao))
    vk = np.zeros((3, nao, nao))
    import copy

    atom = copy.copy(mol)
    aoslice = mol.aoslice_by_atom(ao_loc)
    for ia in range(mol.natm):
        b0, b1, p0, p1 = aoslice[ia]
        atom._bas = mol._bas[b0:b1]
        atm = np.array(copy.copy(atom._atm), dtype=np.int32).flatten()
        bas = np.array(copy.copy(atom._bas), dtype=np.int32).flatten()
        env = np.array(copy.copy(atom._env), dtype=np.float64).flatten()
        vj1, vk1 = Tower.integral.get_bp_hso2e(
            dm0[p0:p1, p0:p1], atom.nao, atm, bas, env
        )
        vj[:, p0:p1, p0:p1] = vj1
        vk[:, p0:p1, p0:p1] = vk1
    return vj, vk


refvj, refvk = get_bp_hso2e_amfi(mol, mf.make_rdm1())
test_result_vj, test_result_vk = get_bp_hso2e_amfi_cpp(mol, mf.make_rdm1())

print(np.abs(refvk - test_result_vk).sum())
print(np.abs(refvj - test_result_vj).sum())
