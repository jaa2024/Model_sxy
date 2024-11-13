from pyscf import gto, mcscf
from pyscf.tools import fcidump

mol = gto.M(atom="n 0 0 -0.55; n 0 0 0.55", basis="sto-3g")
myhf = mol.RHF().run()

ncas, nelecas = (6, 6)
mycas = mcscf.CASCI(myhf, ncas, nelecas)
# mycas.kernel()
fcidump.from_mcscf(mycas, "fcidump.example1")
