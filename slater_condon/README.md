# Full-CI code using slater condon rules

This is a graduate assignment for the Quantum Chemistry course at Shandong University. It is a simple FCI code based on Slater-Condon rules, implemented in C++. The code reads `FCIDUMP` files and `electronic configuration` files. The `FCIDUMP` file is used to obtain electronic integrals, and the `electronic configuration` file is used to perform CASCI or FCI calculations. Below is the file structure:

The `scripts` folder contains a script, `create_config.py`, which generates the electronic configuration file. It requires the user to input:

```python
num_electrons = int(input("Total number of electrons: "))
num_orbitals = int(input("Total number of orbitals: "))
num_docc = int(input("Number of doubly occupied orbitals: "))
num_active = int(input("Number of active orbitals: "))
save_docc = input("Save doubly occupied orbitals? (y/n): ")

```

The `save_docc` parameter indicates whether to preserve doubly occupied orbital information, depending on the FCIDUMP file being used. If the FCIDUMP does not include the inner integrals in the electronic energy, you need to save this information (type "y"). This corresponds to the Kylin keyword: `dumpAll = true`, or the `from_scf` function in `pyscf.tool`. If the FCIDUMP is based on CAS generation, there is no need to save this information (type "n"). This corresponds to the Kylin keyword: `dump = true`, or the `from_mcscf` function in `pyscf.tool`.

The FCIDUMP can be generated using `pyscf` with `creat_fcidump_pyscf.py` or using `Kylin` with `creat_fcidump_kylin.inp`.

In the `example` folder, there are two sets of calculations: one is the FCI calculation for H4 in the STO-3G basis set, and the other is the CASCI calculation for N2 in CAS(6,6).

Thanks to Dr. Song Yinxuan and Ms. Su Jiaao for their help and guidance. For using the powerful DMRG multi-reference program **Kylin**, please visit: [https://kylin-qc.com/](https://kylin-qc.com/)
