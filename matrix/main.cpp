#include "gto/gto.h"

int main() {
  GTO::Mol mol("H 0 0 0;H 0 0 0.7", "sto-3g");
  mol.printAtoms();
  mol.printCintInfo();
  return 0;
}