#ifndef EWALD_LONG_H_
#define EWALD_LONG_H_

#include <cmath>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <omp.h>
#include "type.h"
#include "time_logger.h"

////////////////////////////////////////////////////////////////////////////////
class EwaldLong{
private:
  int natom_;
  const real* q_;
  const real* r_;
  real L_[3];

  real sigma_;
  real sigma2_; // sigma^2
  real k1_; // 2 pi / L
  int nmax_;
  int nmax2_; // nmax^2
  int n_;
  real* A_;
  real* B_;
  real reciprocal_prefactor_; // 4 pi * 2/L^3 -> 1/r
  real self_potential_;

  TimeLogger log_;

public:
  EwaldLong(real sigma, int nmax, int natom, const real* q){
    sigma_ = sigma;
    sigma2_ = sigma_ * sigma_;
    nmax_ = nmax;
    nmax2_ = nmax_ * nmax_;

    natom_ = natom;
    q_ = q;

    self_potential_ = std::inner_product(q_, q_ + natom, q_, 0.0);
    self_potential_ /= sqrt(2 * M_PI) * sigma_;


    // 0 <= nx <= nmax : (nmax+1)
    // -nmax <= ny <= nmax : (2nmax+1)
    // -nmax <= nz <= nmax : (2nmax+1)
    n_ = (nmax_ + 1) * (2 * nmax_ + 1) * (2 * nmax_ + 1);
    A_ = new real[n_];
    B_ = new real[n_];
  }

  ~EwaldLong(){
    if(A_ != nullptr){delete [] A_;}
    if(B_ != nullptr){delete [] B_;}
  }

  static void get_AB(int natom, const real* q, const real* r,
    real kx, real ky, real kz, real* Ak, real* Bk){
    real A = 0;
    real B = 0;

    #pragma omp parallel for reduction(+:A,B)
    for(int j = 0; j < natom; j++){
      const real* rj = &r[3 * j];
      real kr = kx * rj[0] + ky * rj[1] + kz * rj[2];
      A += q[j] * std::cos(kr);
      B += q[j] * std::sin(kr);
    }

    *Ak = A;
    *Bk = B;
  }

  void set_structure_factor(){
    log_.start("structure_factor");

    // 0 <= nx <= nmax : (nmax+1)
    // -nmax <= ny <= nmax : (2nmax+1)
    // -nmax <= nz <= nmax : (2nmax+1)
    const int w = 2 * nmax_ + 1;

    for(int n = 1; n <= nmax_; n++){
      // nx = n
      real kx = k1_ * n;
      int nx = n * w * w;

      for(int ny = -nmax_; ny <= nmax_; ny++){
        real ky = k1_ * ny;
        int nxny = nx + (ny + nmax_) * w;

        for(int nz = -nmax_; nz <= nmax_; nz++){
          real kz = k1_ * nz;
          int nxnynz = nxny + (nz + nmax_);

          get_AB(natom_, q_, r_, kx, ky, kz, &A_[nxnynz], &B_[nxnynz]);
        }
      }

      // nx = 0, ny = n
      kx = 0;
      real ky = k1_ * n;
      int nxny = n * w;

      for(int nz = -nmax_; nz <= nmax_; nz++){
        real kz = k1_ * nz;
        int nxnynz = nxny + (nz + nmax_);

        get_AB(natom_, q_, r_, kx, ky, kz, &A_[nxnynz], &B_[nxnynz]);
      }

      // nx = 0, ny = 0, nz = n
      ky = 0;
      real kz = k1_ * n;
      int nxnynz = n;

      get_AB(natom_, q_, r_, kx, ky, kz, &A_[nxnynz], &B_[nxnynz]);
    }

    log_.stop();
  }

  void set_coordinate(const real* r, const real* L){
    r_ = r;
    std::copy_n(L, 3, L_);

//TODO
    if(L[1] != L[0] || L[2] != L[0]){
      throw std::invalid_argument(
        "Rectangular cell is not supported in EwaldLong");
    }

//TODO
    real L_inv = 1.0 / L_[0];
    reciprocal_prefactor_ = 2.0 * L_inv * L_inv * L_inv;
    reciprocal_prefactor_ *= 4.0 * M_PI;
    k1_ = 2.0 * M_PI * L_inv;
  }

  static void add_value_gradient(real A, real B,
    real kr, real k2, real sigma2, const real* k,
    real* value, real* gradient){
    real c = std::cos(kr);
    real s = std::sin(kr);
    real a = std::exp(- k2 * sigma2 / 2.0) / k2;

    *value += a * (A * c + B * s);

    real b = a * (- A * s + B * c);
    gradient[0] += b * k[0];
    gradient[1] += b * k[1];
    gradient[2] += b * k[2];
  }

  // (r[0],r[1],r[2]) = (x,y,z)
  real get_value_gradient(const real* r, real* gradient){
    // 0 <= nx <= nmax : (nmax+1)
    // -nmax <= ny <= nmax : (2nmax+1)
    // -nmax <= nz <= nmax : (2nmax+1)
    const int w = 2 * nmax_ + 1;
    real value = 0;
    real k[3];

    std::fill_n(gradient, 3, 0);

    for(int n = 1; n <= nmax_; n++){
      // nx = n
      k[0] = k1_ * n;
      real kxx = k[0] * r[0];
      int nx = n * w * w;
      real nx2 = n * n;

      for(int ny = -nmax_; ny <= nmax_; ny++){
        k[1] = k1_ * ny;
        int nxny = nx + (ny + nmax_) * w;
        real kxxkyy = kxx + k[1] * r[1];
        int nx2ny2 = nx2 + ny * ny;

        for(int nz = -nmax_; nz <= nmax_; nz++){
          int n2 = nx2ny2 + nz * nz;

          if(n2 <= nmax2_){
            k[2] = k1_ * nz;
            int nxnynz = nxny + (nz + nmax_);
            real kr = kxxkyy + k[2] * r[2];
            real k2 = k1_ * k1_ * n2;

            add_value_gradient(
              A_[nxnynz], B_[nxnynz], kr, k2, sigma2_, k, &value, gradient);
          }
        }
      }

      // nx = 0, ny = n
      k[0] = 0;
      k[1] = k1_ * n;
      real kxxkyy = k[1] * r[1];
      int nxny = n * w;
      int nx2ny2 = n * n;

      for(int nz = -nmax_; nz <= nmax_; nz++){
        int n2 = nx2ny2 + nz * nz;

        if(n2 <= nmax2_){
          k[2] = k1_ * nz;
          int nxnynz = nxny + (nz + nmax_);
          real kr = kxxkyy + k[2] * r[2];
          real k2 = k1_ * k1_ * n2;

          add_value_gradient(
            A_[nxnynz], B_[nxnynz], kr, k2, sigma2_, k, &value, gradient);
        }
      }

      // nx = 0, ny = 0, nz = n
      int n2 = n * n;

      if(n2 <= nmax2_){
        k[0] = 0;
        k[1] = 0;
        k[2] = k1_ * n;
        int nxnynz = n;
        real kr = k[2] * r[2];
        real k2 = k1_ * k1_ * n2;

        add_value_gradient(A_[nxnynz], B_[nxnynz],
          kr, k2, sigma2_, k, &value, gradient);
      }
    }

    value *= reciprocal_prefactor_;
    gradient[0] *= reciprocal_prefactor_;
    gradient[1] *= reciprocal_prefactor_;
    gradient[2] *= reciprocal_prefactor_;

    return value;
  }

  void get_value_gradient(int ntarget, const real* rtarget,
    real* value, real* gradient){
    #pragma omp parallel for
    for(int i = 0; i < ntarget; i++){
      value[i] = get_value_gradient(&rtarget[3 * i], &gradient[3 * i]);
    }
  }

  real get_potential_force(real* force){
    log_.start("long_interaction");

//TODO
    real* value = new real[natom_];

    get_value_gradient(natom_, r_, value, force);

    real potential = 0;

    for(int i = 0; i < natom_; i++){
      potential += q_[i] * value[i];
    }

    for(int i = 0; i < natom_; i++){
      real* fi = &force[3 * i];
      fi[0] *= - q_[i];
      fi[1] *= - q_[i];
      fi[2] *= - q_[i];
    }

    potential *= 0.5;
    potential -= self_potential_;

//TODO
    delete [] value;

    log_.stop();

    return potential;
  }

  void print_parameter(std::ostream& o) const{
    o << "ewald_long parameter:" << std::endl;
    o << "sigma " << sigma_ << std::endl;
    o << "nmax " << nmax_ << std::endl;
  }

};

#endif // EWALD_LONG_H_
