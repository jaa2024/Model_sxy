#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <tuple>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

// Class to handle quantum chemistry integrals (one and two electron integrals)
class Integral {
public:
  // Constructor that reads integral data from a file
  Integral(const std::string &filename) {
    norb = 0;
    coreE = 0;

    std::ifstream file(filename);
    if (file.is_open()) {
      std::vector<std::string> lines;
      std::string line;

      // Read all lines from file
      while (std::getline(file, line)) {
        lines.emplace_back(line);
      }

      file.close();

      // Parse the number of orbitals from first line
      std::istringstream iss(lines[0]);
      std::string part;
      std::vector<std::string> parts;
      while (iss >> part) {
        parts.emplace_back(part);
      }
      norb = std::stoi(parts[2].substr(0, parts[2].length() - 1));

      // Initialize one-electron integral matrix
      int1e = Eigen::MatrixXd::Zero(norb, norb);

      // Initialize two-electron integral tensor
      int2e = Eigen::Tensor<double, 4>(norb, norb, norb, norb);
      int2e.setZero();

      // Parse integral values from file
      for (size_t i = 4; i < lines.size(); ++i) {
        std::istringstream iss2(lines[i]);
        std::vector<std::string> parts2;
        while (iss2 >> part) {
          parts2.emplace_back(part);
        }

        double val = std::stod(parts2[0]);
        int p = std::stoi(parts2[1]) - 1;
        int q = std::stoi(parts2[2]) - 1;
        int r = std::stoi(parts2[3]) - 1;
        int s = std::stoi(parts2[4]) - 1;

        // Handle different types of integrals
        if (r == -1 && s == -1) {
          if (p == -1 && q == -1) {
            coreE = val; // Nuclear repulsion energy
          } else {
            // One-electron integrals
            int1e(p, q) = val;
            int1e(q, p) = val;
          }
        } else {
          // Two-electron integrals with symmetry
          int2e(p, q, r, s) = val;
          int2e(p, q, s, r) = val;
          int2e(q, p, r, s) = val;
          int2e(q, p, s, r) = val;
          int2e(r, s, p, q) = val;
          int2e(r, s, q, p) = val;
          int2e(s, r, p, q) = val;
          int2e(s, r, q, p) = val;
        }
      }
    } else {
      std::cerr << "could not open: " << filename << std::endl;
    }
  }

  // Get one-electron integral value
  auto h1e(int p, int q) const -> double { return int1e(p, q); }

  // Get two-electron integral value
  auto h2e(int p, int q, int r, int s) const -> double {
    return int2e(p, q, r, s);
  }

  // Get nuclear repulsion energy
  auto getCoreE() const -> double { return coreE; }

private:
  int norb;                       // Number of orbitals
  double coreE;                   // Nuclear repulsion energy
  Eigen::MatrixXd int1e;          // One-electron integrals
  Eigen::Tensor<double, 4> int2e; // Two-electron integrals
};

// Class to represent Slater determinants
class Det {
public:
  // Constructor that creates determinant from occupation string
  // '0': empty orbital
  // 'u': alpha electron
  // 'd': beta electron
  // '2': doubly occupied orbital
  Det(const std::string &str) {
    alpha.clear();
    beta.clear();
    na = 0;
    nb = 0;
    nelec = 0;
    norb = 0;

    for (auto c : str) {
      if (c == '0') {
        alpha.emplace_back(0);
        beta.emplace_back(0);
      } else if (c == 'u') {
        alpha.emplace_back(1);
        beta.emplace_back(0);
        na += 1;
      } else if (c == 'd') {
        alpha.emplace_back(0);
        beta.emplace_back(1);
        nb += 1;
      } else if (c == '2') {
        alpha.emplace_back(1);
        beta.emplace_back(1);
        na += 1;
        nb += 1;
      }
    }

    nelec = na + nb;
    norb = alpha.size();
  }

  // Getters for various properties
  auto getNa() const -> int { return na; }
  auto getNb() const -> int { return nb; }
  auto getNelec() const -> int { return nelec; }
  auto getNorb() const -> int { return norb; }

  // Convert determinant to string representation
  auto to_string() const -> std::string {
    std::stringstream ss;
    for (auto i = 0; i < norb; ++i) {
      if (alpha[i] == 1 && beta[i] == 0) {
        ss << 'u';
      } else if (alpha[i] == 0 && beta[i] == 1) {
        ss << 'd';
      } else if (alpha[i] == 1 && beta[i] == 1) {
        ss << '2';
      } else if (alpha[i] == 0 && beta[i] == 0) {
        ss << '0';
      }
    }
    return ss.str();
  }

  // Count electrons between orbital p and q
  auto count_elec(int p, int q, bool alpha = true) const -> int {
    if (p > q) {
      std::swap(p, q);
    }

    auto count{0};
    const std::vector<int> &list = alpha ? this->alpha : this->beta;

    for (auto i = p + 1; i < q; ++i) {
      if (list[i] == 1) {
        count += 1;
      }
    }

    return count;
  }

  // Compare two occupation lists and return occupied and virtual orbitals
  auto compare(const std::vector<int> &list1,
               const std::vector<int> &list2) const
      -> std::tuple<std::vector<int>, std::vector<int>> {
    std::vector<int> occ;
    std::vector<int> vir;

    for (auto i = 0; i < norb; ++i) {
      if (list1[i] != list2[i]) {
        if (list1[i] == 1) {
          occ.emplace_back(i);
        } else {
          vir.emplace_back(i);
        }
      }
    }

    return std::make_tuple(occ, vir);
  }

  // Calculate diagonal matrix element (Hii)
  auto Hii(const Integral &integral) const -> double {
    double val{0};
    // Sum over all orbitals
    for (int p = 0; p < norb; ++p) {
      // Contribution from alpha electrons
      if (alpha[p] == 1) {
        val += integral.h1e(p, p);

        for (int q = 0; q < norb; ++q) {
          if (alpha[q] == 1) {
            val += 0.5 * (integral.h2e(p, p, q, q) - integral.h2e(p, q, p, q));
          }
          if (beta[q] == 1) {
            val += 0.5 * integral.h2e(p, p, q, q);
          }
        }
      }
      // Contribution from beta electrons
      if (beta[p] == 1) {
        val += integral.h1e(p, p);

        for (int q = 0; q < norb; ++q) {
          if (beta[q] == 1) {
            val += 0.5 * (integral.h2e(p, p, q, q) - integral.h2e(p, q, p, q));
          }
          if (alpha[q] == 1) {
            val += 0.5 * integral.h2e(p, p, q, q);
          }
        }
      }
    }
    return val;
  }

  // Calculate sign based on number of electrons
  auto sign(int nelec) const -> double { return std::pow(-1, nelec); }

  // Calculate off-diagonal matrix element (Hij)
  auto Hij(const Det &det, const Integral &integral) const -> double {
    // Compare alpha and beta occupations between determinants
    auto [occa, vira] = compare(alpha, det.alpha);
    auto [occb, virb] = compare(beta, det.beta);

    int ndiffa = occa.size();
    int ndiffb = occb.size();
    int ndiff = ndiffa + ndiffb;

    // Slater-Condon rules implementation
    if (ndiff > 2) {
      return 0;
    } else if (ndiff == 1) { // Single excitation
      if (ndiffa == 1) {     // Alpha excitation
        int p = occa[0];
        int q = vira[0];
        double h1 = integral.h1e(p, q);
        double h2 = 0;
        for (int r = 0; r < norb; ++r) {
          if (alpha[r] == 1) {
            h2 += integral.h2e(p, q, r, r) - integral.h2e(p, r, r, q);
          }
          if (beta[r] == 1) {
            h2 += integral.h2e(p, q, r, r);
          }
        }

        return sign(count_elec(p, q, true)) * (h1 + h2);
      } else if (ndiffb == 1) { // Beta excitation
        int p = occb[0];
        int q = virb[0];
        double h1 = integral.h1e(p, q);
        double h2 = 0;
        for (int r = 0; r < norb; ++r) {
          if (beta[r] == 1) {
            h2 += integral.h2e(p, q, r, r) - integral.h2e(p, r, r, q);
          }
          if (alpha[r] == 1) {
            h2 += integral.h2e(p, q, r, r);
          }
        }
        return sign(count_elec(p, q, false)) * (h1 + h2);
      } else {
        return 0;
      }
    } else if (ndiff == 2) { // Double excitation
      if (ndiffa == 2) {     // Double alpha excitation
        int p = occa[0];
        int r = occa[1];
        int q = vira[0];
        int s = vira[1];
        if (p > r) {
          std::swap(p, r);
        }
        if (q > s) {
          std::swap(q, s);
        }

        double val = sign(count_elec(p, q, true));
        val *= sign(det.count_elec(r, s, true));
        val *= (integral.h2e(p, q, r, s) - integral.h2e(p, s, r, q));

        return val;
      } else if (ndiffb == 2) { // Double beta excitation
        int p = occb[0];
        int r = occb[1];
        int q = virb[0];
        int s = virb[1];
        if (p > r) {
          std::swap(p, r);
        }
        if (q > s) {
          std::swap(q, s);
        }

        double val = sign(count_elec(p, q, false));
        val *= sign(det.count_elec(r, s, false));
        val *= (integral.h2e(p, q, r, s) - integral.h2e(p, s, r, q));

        return val;
      } else if (ndiffa == 1 && ndiffb == 1) { // Alpha-beta excitation
        int p = occa[0];
        int q = vira[0];
        int r = occb[0];
        int s = virb[0];
        double val = sign(count_elec(p, q, true));
        val *= sign(count_elec(r, s, false));
        val *= integral.h2e(p, q, r, s);
        return val;
      } else {
        return 0;
      }
    } else {
      return 0;
    }
  }

private:
  std::vector<int> alpha; // Alpha electron occupation
  std::vector<int> beta;  // Beta electron occupation
  int na;                 // Number of alpha electrons
  int nb;                 // Number of beta electrons
  int nelec;              // Total number of electrons
  int norb;               // Number of orbitals
};

// Read determinants from file
auto readDets(const std::string &fileName) -> std::vector<Det> {
  std::vector<Det> dets;
  std::ifstream file(fileName);
  if (file.is_open()) {
    std::string line;
    while (std::getline(file, line)) {
      dets.emplace_back(Det(line));
    }
    file.close();
  } else {
    std::cerr << "could not open: " << fileName << std::endl;
  }

  return dets;
}
auto readH(const std::string &fileName) -> Eigen::MatrixXd {
  std::ifstream file(fileName);
  if (file.is_open()) {
    std::string line;

    std::getline(file, line);
    std::istringstream issFirst(line);
    int dim;
    issFirst >> dim;

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dim, dim);

    while (std::getline(file, line)) {
      std::istringstream iss(line);
      int row, col;
      double val;

      iss >> row;
      iss >> col;
      iss >> val;

      H(row, col) = val;
      H(col, row) = val;
    }

    file.close();

    return H;
  } else {
    std::cerr << "could not open: " << fileName << std::endl;
    return Eigen::MatrixXd();
  }
}

auto main() -> int {
  auto start = std::chrono::system_clock::now();
  auto dets = readDets("../example/n2_fci/electron_configurations.txt");
  auto integral = Integral("../example/n2_fci/fcidump.example1");

  auto ndet = dets.size();

  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(ndet, ndet);

#pragma omp parallel for schedule(static)
  for (auto i = 0; i < ndet; i++) {
    H(i, i) = dets[i].Hii(integral);

    for (auto j = i + 1; j < ndet; j++) {
      H(i, j) = dets[i].Hij(dets[j], integral);
      H(j, i) = H(i, j);
    }
  }

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H);
  Eigen::VectorXd eigenvalues = solver.eigenvalues();

  auto E = eigenvalues(0) + integral.getCoreE();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::system_clock::now() - start);
  std::cout << "E = " << std::fixed << std::setprecision(12) << E << std::endl;
  std::cout << "time = " << std::fixed << std::setprecision(12)
            << duration.count() << "s" << std::endl;
  // std::cout << "H = " << std::endl << H << std::endl;
  // auto H_benchmark = readH("../doc/H-h4.txt");

  // for (auto i = 0; i < ndet; i++) {
  //   for (auto j = 0; j < ndet; j++) {
  //     if (std::abs(H(i, j) - H_benchmark(i, j)) > 1e-8) {
  //       std::cout << "i = " << i << "\n"
  //                 << " j = " << j << "\n"
  //                 << "det1 = " << dets[i].to_string() << "\n"
  //                 << "det1 = " << dets[j].to_string() << "\n"
  //                 << "H" << H(i, j) << "\n"
  //                 << "H_benchmark" << H_benchmark(i, j) << "\n";
  //     }
  //   }
  // }
  return 0;
}