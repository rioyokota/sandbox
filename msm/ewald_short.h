#ifndef EWALD_SHORT_H_
#define EWALD_SHORT_H_

#include <algorithm>
#include "system.h"
#include "time_logger.h"

////////////////////////////////////////////////////////////////////////////////
class EwaldShort{
private:
  int natom_;
  const real* q_;
  const real* r_;
  real L_[3];
  real eps_; // For x < eps, erf(a*r)/r is evalueted by Taylor expansion
  real taylor_prefactor_;
  real sigma_;
  real cutoff_;
  real sigma2_; // sigma^2
  real alpha_; // 1 / (sqrt(2) * sigma)
  real beta_; // (1 / (sqrt(2) sigma)) (2 / sqrt(pi))

  TimeLogger log_;

public:
  EwaldShort(real sigma, real cutoff, int natom, const real* q){
    sigma_ = sigma;
    sigma2_ = sigma_ * sigma_;
    cutoff_ = cutoff;
    alpha_ = 1.0 / (std::sqrt(2) * sigma_);
    beta_ = alpha_ * 2.0 / std::sqrt(M_PI);
    eps_ = 1e-5;
    taylor_prefactor_ = std::sqrt(2.0 / M_PI) / sigma_;
    natom_ = natom;
    q_ = q;
  }

  void set_coordinate(const real* r, const real* L){
    r_ = r;
    std::copy_n(L, 3, L_);
  }

  real get_short_potential_force(real r, const real* dr, real* force) const{
    real x = alpha_ * r;
    real r_inv = 1.0 / r;
    real potential = r_inv * std::erfc(x);

    real fr = r_inv * (potential + beta_ * std::exp(- x * x));
    real fr_r = r_inv * fr; // F(r)/r

    force[0] = fr_r * dr[0]; // F(r)*(x/r)
    force[1] = fr_r * dr[1];
    force[2] = fr_r * dr[2];

    return potential;
  }

//TODO rm
  // erf(r / (sqrt(2) * sigma)) / r
  static real get_ewald_kernel(real r, real sigma){
    if(sigma == 0){
      return 1.0 / r;
    }
    else{
      real x = r / (std::sqrt(2) * sigma);

      if(x > 1e-5){
        return std::erf(x) / r;
      }
      else{
        return (std::sqrt(2.0 / M_PI) / sigma) * (1.0 - (1.0 / 3.0) * x * x);
      }
    }
  }

  real get_potential_force(real* force){
    log_.start("short_interaction");

    real cutoff2 = cutoff_ * cutoff_;
//TODO
    real* value = new real[natom_];
std::fill_n(value, natom_, 0);

    #pragma omp parallel for
    for(int i = 0; i < natom_; i++){
      const real* ri = &r_[3 * i];
      real p = 0;
      real fx = 0;
      real fy = 0;
      real fz = 0;

//TODO j = i + 1
      for(int j = 0; j < natom_; j++){
        if(j == i){
          continue;
        }

        const real* rj = &r_[3 * j];

        // minimum image convention
        real dr[3];
        PeriodicBoundary::get_minimum_vector(rj, ri, L_, dr);
        real r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];

        if(r2 <= cutoff2){
          real r = std::sqrt(r2);
          real fj[3];
          p += q_[j] * get_short_potential_force(r, dr, fj);
          fx += q_[j] * fj[0];
          fy += q_[j] * fj[1];
          fz += q_[j] * fj[2];
        }
      }

//TODO
      value[i] = 0.5 * q_[i] * p;
      real* fi = &force[3 * i];
      fi[0] = q_[i] * fx;
      fi[1] = q_[i] * fy;
      fi[2] = q_[i] * fz;
    }

    real potential = 0;

    for(int i = 0; i < natom_; i++){
      potential += value[i];
    }

    log_.stop();

    return potential;
  }

  void print_parameter(std::ostream& o) const{
    o << "ewald_short parameter:" << std::endl;
    o << "sigma " << sigma_ << std::endl;
    o << "cutoff " << cutoff_ << std::endl;
  }

};

#endif // EWALD_SHORT_H_
