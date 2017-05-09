#ifndef SYSTEM_H_
#define SYSTEM_H_

#include "type.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <omp.h>

////////////////////////////////////////////////////////////////////////////////
class System{
private:
  int natom_;
  real* q_;
  real* r_;
  real L_[3];

public:
  System(){
    natom_ = 0;
    q_ = nullptr;
    r_ = nullptr;
    std::fill_n(L_, 3, 0);
  }

  ~System(){
    if(q_ != nullptr){delete [] q_;}
    if(r_ != nullptr){delete [] r_;}
  }

  static void read_vector(const std::string& file, std::vector<real>& x){
    std::ifstream ifs(file.c_str());

    if(!ifs){
      throw std::invalid_argument(file + " can not be opened.");
    }

    std::string line;

    while(std::getline(ifs, line)){
      std::istringstream s(line);
      real xi;

      while(s >> xi){
        x.push_back(xi);
      }
    }
  }

  void read_charge(const std::string& file){
    std::vector<real> q;

    read_vector(file, q);

    if(natom_ != 0 && natom_ != q.size()){
      std::invalid_argument("Atom number is incorrect.");
    }

    natom_ = q.size();

    if(q_ == nullptr){
      q_ = new real[natom_];
    }

    std::copy(q.begin(), q.end(), q_);
  }

  void read_coordinate(const std::string& file){
    std::vector<real> r;
    std::vector<real> L;

    read_vector(file, r);

    int start = r.size() - 3;

    L.push_back(r[start]);
    L.push_back(r[start + 1]);
    L.push_back(r[start + 2]);

    r.pop_back();
    r.pop_back();
    r.pop_back();

    natom_ = r.size() / 3;

    if(natom_ != 0 && r.size() != 3 * natom_){
      throw std::invalid_argument("Atom number is incorrect.");
    }

    if(r_ == nullptr){
      r_ = new real[3 * natom_];
    }

    std::copy(r.begin(), r.end(), r_);
    std::copy(L.begin(), L.end(), L_);
  }

  void convert_unit_cell(){
    const real L_inv[] = {1.0 / L_[0], 1.0 / L_[1], 1.0 / L_[2]};

    #pragma omp parallel for
    for(int i = 0; i < natom_; i++){
      real* ri = &r_[3 * i];

      for(int j = 0; j < 3; j++){
        ri[j] -= std::floor(ri[j] * L_inv[j]) * L_[j];
      }
    }
  }

  int get_atom_number() const{
    return natom_;
  }

  const real* get_charge() const{
    return q_;
  }

  const real* get_coordinate() const{
    return r_;
  }

  const real* get_unit_cell_length() const{
    return L_;
  }

  void print_parameter(std::ostream& o) const{
    o << "atom " << natom_ << std::endl;
    o << "unit_cell_length"
      << " " << L_[0]
      << " " << L_[1]
      << " " << L_[2]
      << std::endl;
  }

};

////////////////////////////////////////////////////////////////////////////////
class PeriodicBoundary{
public:
  static void get_minimum_vector(const real* r1, const real* r2,
    const real* L, real* dr){
    // minimum image convention
    const real L_inv[] = {1.0 / L[0], 1.0 / L[1], 1.0 / L[2]};

    for(int j = 0; j < 3; j++){
      dr[j] = (r2[j] - r1[j]) * L_inv[j];
      dr[j] = (dr[j] - std::round(dr[j])) * L[j];
    }
  }

};

#endif // SYSTEM_H_
