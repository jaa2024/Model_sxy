# RHF code for teaching

Here is the documentation for the code:

This is an introductory task for new graduate students joining the research group. The task involves using the `gto` module of PySCF to construct a `mol` class, which leverages information from the `mol` class to call `libcint` for the computation of one- and two-electron integrals. The `RHF` class implements a basic Restricted Hartree-Fock (RHF) calculation, with a partial consideration of electron integral symmetries.

The `build.py` script enables automatic compilation and linking with the `libcint.so` library. It only supports Linux and macOS systems (due to PySCF's compatibility).

This code requires the system to have a C language compiler such as GCC and CMake installed. It also requires PySCF's `gto` module to create molecular information, as well as NumPy and SciPy for matrix storage and related operations.

To run, use the following command:

```bash
python rhf_full_sxy.py
```
