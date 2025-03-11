#include "gto/gto.h"
#include <algorithm>
#include <fstream>
#include <regex>
#include <set>

namespace GTO {
Mol::Mol(const std::string &xyz, const std::string &basis, const int spin,
         const int charge)
    : xyz_info_(xyz), basis_info_(basis), spin_(spin), charge_(charge) {
  parseXYZ(xyz_info_);
  parseBasis(basis_info_);
  setupCintInfo();
}

void Mol::nuclear_repulsion_() {
  for (std::size_t i = 0; i < atoms_.size(); i++)
    for (std::size_t j = i + 1; j < atoms_.size(); j++) {
      auto xij = atoms_[i].x * ANSTROM_TO_BOHR - atoms_[j].x * ANSTROM_TO_BOHR;
      auto yij = atoms_[i].y * ANSTROM_TO_BOHR - atoms_[j].y * ANSTROM_TO_BOHR;
      auto zij = atoms_[i].z * ANSTROM_TO_BOHR - atoms_[j].z * ANSTROM_TO_BOHR;
      auto r2 = xij * xij + yij * yij + zij * zij;
      auto r = sqrt(r2);
      nuc_rep_ += atoms_[i].Z * atoms_[j].Z / r;
    }
}

void Mol::nelectron_() {
  nelec_[2] = charge_;
  for (std::size_t i = 0; i < atoms_.size(); i++) {
    nelec_[2] += atoms_[i].Z;
  }
  nelec_[0] = (nelec_[2] + spin_) / 2;
  nelec_[1] = nelec_[2] - nelec_[0];
}

void Mol::parseXYZ(const std::string &xyz) {
  std::istringstream iss(xyz);
  std::string atomInfo;

  while (std::getline(iss, atomInfo, ';')) {

    atomInfo = atomInfo.substr(atomInfo.find_first_not_of(" "),
                               atomInfo.find_last_not_of(" ") -
                                   atomInfo.find_first_not_of(" ") + 1);

    std::istringstream atomStream(atomInfo);
    std::string atomSymbol;
    double x, y, z;

    atomStream >> atomSymbol >> x >> y >> z;
    if (atomStream.fail()) {
      throw std::invalid_argument("Failed to parse atom line: " + atomInfo);
    }

    if (!atomSymbol.empty()) {
      if (atomSymbol.size() == 1) {
        std::transform(atomSymbol.begin(), atomSymbol.end(), atomSymbol.begin(),
                       ::toupper);
      } else if (atomSymbol.size() == 2) {

        atomSymbol[0] = std::toupper(atomSymbol[0]);
        atomSymbol[1] = std::tolower(atomSymbol[1]);
      }
    }

    auto it = ELEMENT_TABLE.find(atomSymbol);
    if (it == ELEMENT_TABLE.end()) {
      throw std::invalid_argument("Unknown element symbol: " + atomSymbol);
    }
    auto Z = it->second;

    atoms_.push_back({atomSymbol, Z, x, y, z});
  }
  nuclear_repulsion_();
  nelectron_();
}

void Mol::parseBasis(const std::string &basis) {
  basis_name_ = basis;
  std::transform(basis_name_.begin(), basis_name_.end(), basis_name_.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  // double *
  basis_name_ = std::regex_replace(basis_name_, std::regex("\\*\\*"), "_d_p_");
  // single *
  basis_name_ = std::regex_replace(basis_name_, std::regex("\\*"), "_d_");
}

void Mol::setupCintInfo() {
  // 初始化 atm 和 env 的大小
  info_.natm = static_cast<int>(atoms_.size());
  info_.atm.resize(ATM_SLOTS * info_.natm); // atm 是二维数组，展平为一维

  info_.env.resize(20, 0.0);

  int env_index = PTR_ENV_START;
  for (int i = 0; i < info_.natm; ++i) {
    const auto &atom = atoms_[i];

    info_.atm[i * ATM_SLOTS + 0] = atom.Z;

    info_.atm[i * ATM_SLOTS + 1] = env_index;
    info_.env.push_back(atom.x * ANSTROM_TO_BOHR);
    info_.env.push_back(atom.y * ANSTROM_TO_BOHR);
    info_.env.push_back(atom.z * ANSTROM_TO_BOHR);
    info_.env.push_back(0.0);
    env_index += 4;

    info_.atm[i * ATM_SLOTS + 2] = 1;

    info_.atm[i * ATM_SLOTS + 3] = env_index - 1;

    info_.atm[i * ATM_SLOTS + 4] = 0;
    info_.atm[i * ATM_SLOTS + 5] = 0;
  }

  std::set<int> unique_Z;
  for (const auto &a : atoms_) {
    unique_Z.insert(a.Z);
  }

  std::vector<int> sorted_Z(unique_Z.begin(), unique_Z.end());
  std::sort(sorted_Z.begin(), sorted_Z.end());

  std::vector<std::pair<std::string, int>> result;
  for (int Z : sorted_Z) {
    auto it = ELEMENT_TABLE_REVERSED.find(Z);
    if (it != ELEMENT_TABLE_REVERSED.end()) {
      result.emplace_back(it->second, Z);
    } else {
      result.emplace_back("Unknown", Z);
    }
  }

  std::string sep;
#ifdef _WIN32
  sep = "\\";
#else
  sep = "/";
#endif
  std::string basisfile = "share" + sep + "basis" + sep + basis_name_ + ".g94";
  std::string line, am;
  std::fstream fin(basisfile, std::ios::in);
  int idxe, idxc;
  if (fin.good()) {
    std::vector<std::string> wl;
    int l, ncf1, cf1, npf, pf;

    for (auto e : result) {
      do // Skip the beginning for basisfile
      {
        getline(fin, line);
        line = line.erase(line.find_last_not_of("\r\n") + 1);
      } while (line != "****");
      fin.clear();
      fin.seekg(0, std::ios::beg);
      std::vector<Shell> shl;

      do // Each line begins with symb    0
      {
        getline(fin, line);
        line = line.erase(line.find_last_not_of("\r\n") + 1);
        if (fin.eof()) {
          std::string err = basisfile + " is broken!!!!";
          throw std::runtime_error(err);
        }

      } while (line != e.first + "     0 ");

      getline(fin, line); // S/SP 3/6 1.0
      line = line.erase(line.find_last_not_of("\r\n") + 1);
      do {
        wl = split(line);
        am = wl[0];
        npf = stoi(wl[1]);

        if (am == "SP") {
          linalg::Matrix<double> ec(npf, 2), ec1(npf, 2);
          for (pf = 0; pf < npf; ++pf) {
            getline(fin, line);
            line = line.erase(line.find_last_not_of("\r\n") + 1);
            wl = split(line);
            std::replace(wl[1].begin() + 8, wl[1].end(), 'D', 'E');
            std::replace(wl[2].begin() + 8, wl[2].end(), 'D', 'E');
            ec(pf, 0) = static_cast<double>(stod(wl[0]));
            ec(pf, 1) = static_cast<double>(stod(wl[1]));
            ec1(pf, 0) = static_cast<double>(stod(wl[0]));
            ec1(pf, 1) = static_cast<double>(stod(wl[2]));
          }

          shlNormalize_(0, ec);
          idxe = info_.env.size();
          std::vector<double> exp;
          for (int i = 0; i < ec.rows(); ++i) {
            exp.push_back(ec(i, 0)); // 获取第 i 行，第 0 列的值
          }
          info_.env.insert(info_.env.end(), exp.begin(), exp.end());
          idxc = info_.env.size();
          std::vector<double> con;
          for (auto i = 0; i < ec.rows(); ++i) {
            con.push_back(ec(i, 1));
          }
          info_.env.insert(info_.env.end(), con.begin(), con.end());

          shl.emplace_back(Shell{am, {0, npf, 1, 0, idxe, idxc}});

          shlNormalize_(1, ec1);
          idxe = info_.env.size();
          std::vector<double> exp1;
          for (auto i = 0; i < ec1.rows(); ++i) {
            exp1.push_back(ec1(i, 0));
          }
          info_.env.insert(info_.env.end(), exp1.begin(), exp1.end());
          int idxc = info_.env.size();
          std::vector<double> con1;
          for (auto i = 0; i < ec1.rows(); ++i) {
            con1.push_back(ec1(i, 1));
          }
          info_.env.insert(info_.env.end(), con1.begin(), con1.end());

          shl.emplace_back(Shell({am, {1, npf, 1, 0, idxe, idxc}}));
        } else {
          l = AML.at(am);
          getline(fin, line);
          line = line.erase(line.find_last_not_of("\r\n") + 1);
          wl = split(line);

          ncf1 = wl.size();
          linalg::Matrix<double> ec(npf, ncf1);
          for (cf1 = 0; cf1 < ncf1; ++cf1) {
            std::replace(wl[cf1].begin() + 8, wl[cf1].end(), 'D', 'E');
            ec(0, cf1) = stod(wl[cf1]);
          }
          for (pf = 1; pf < npf; ++pf) {
            getline(fin, line);
            wl = split(line);
            line = line.erase(line.find_last_not_of("\r\n") + 1);
            for (cf1 = 0; cf1 < ncf1; ++cf1) {
              std::replace(wl[cf1].begin() + 8, wl[cf1].end(), 'D', 'E');
              ec(pf, cf1) = stod(wl[cf1]);
            }
          }

          shlNormalize_(l, ec);
          idxe = info_.env.size();

          std::vector<double> exp;
          for (auto i = 0; i < ec.rows(); ++i) {
            exp.push_back(ec(i, 0));
          }
          info_.env.insert(info_.env.end(), exp.begin(), exp.end());
          idxc = info_.env.size();
          std::vector<double> con;
          for (auto i = 0; i < ec.rows(); ++i) {
            con.push_back(ec(i, 1));
          }
          info_.env.insert(info_.env.end(), con.begin(), con.end());

          shl.emplace_back(Shell{am, {l, npf, ncf1 - 1, 0, idxe, idxc}});
        }

        getline(fin, line); // S/SP 3/6 1.0
        line = line.erase(line.find_last_not_of("\r\n") + 1);
      } while (line != "****");

      dshl_[e.first] = shl;
    }

    fin.close();
  } else {
    std::string err = basisfile + " not exist!";
    throw std::runtime_error(err);
  }

  for (auto i = 0; i < info_.natm; i++) {
    auto sym = atoms_[i].symbol;
    auto shls = dshl_[sym];
    for (auto &shl : shls) {
      info_.bas.push_back(i);
      info_.bas.insert(info_.bas.end(), shl.bas_info.begin(),
                       shl.bas_info.end());
      info_.bas.push_back(0);
    }
  }
  info_.nbas = info_.bas.size() / BAS_SLOTS;
}

std::vector<std::string> Mol::split(const std::string &_str,
                                    const std::string &_flag) {
  std::vector<std::string> result;
  std::string str = _str + _flag;
  auto size = _flag.size();
  std::string sub;

  for (auto i = 0; i < str.size();) {
    auto p = str.find(_flag, i);
    sub = str.substr(i, p - i);
    if (sub != "") {
      result.emplace_back(sub);
    }
    i = p + size;
  }
  return result;
}
void Mol::shlNormalize_(int l, linalg::Matrix<double> &shl) {

  for (int i = 0; i < shl.rows(); i++) {
    shl(i, 1) *= CINTgto_norm(l, shl(i, 0));
  }
}

void Mol::printAtoms() const {
  fmt::print("Atoms:{}:\n", atoms_.size());
  fmt::print("Spin: {}, Charge: {}\n", spin_, charge_);
  for (const auto &atom : atoms_) {
    fmt::print("Atom: Z={}, x={}, y={}, z={}\n", atom.Z, atom.x, atom.y,
               atom.z);
  }
}

void Mol::printCintInfo() const {
  fmt::print("natm = {}\n", info_.natm);
  fmt::print("nbas = {}\n", info_.nbas);
  // atm
  fmt::print("\natm:\n");
  for (auto i = 0; i < info_.natm; i++) {
    for (auto j = 0; j < ATM_SLOTS; j++) {
      fmt::print("{:>4}", info_.atm[i * ATM_SLOTS + j]);
    }
    fmt::print("\n");
  }
  // bas
  fmt::print("\nbas:\n");
  for (auto i = 0; i < info_.nbas; i++) {
    for (auto j = 0; j < BAS_SLOTS; j++) {
      fmt::print("{:>4}", info_.bas[i * BAS_SLOTS + j]);
    }
    fmt::print("\n");
  }
  // env
  fmt::print("\nenv:\n");
  for (auto i = 0; i < info_.env.size(); i++) {
    fmt::print("{:10.4f}", info_.env[i]);
    if ((i + 1) % 5 == 0) {
      fmt::print("\n");
    }
  }
  fmt::print("\n");
}

cint_info Mol::get_cint_info() const { return info_; }

double Mol::get_nuc_rep() const { return nuc_rep_; }

const std::array<int, 3> &Mol::get_nelec() const { return nelec_; }

} // namespace GTO