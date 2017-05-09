#ifndef COULOMB_COMMAND_H_
#define COULOMB_COMMAND_H_

#include "msm_long.h"
#include "ewald_short.h"
#include "ewald_long.h"

////////////////////////////////////////////////////////////////////////////////
class CoulombCommand{
private:
  real sigma_;
  real cutoff_;
  int nmax_;
  int p_;
  int quadrature_;
  int level_;
  bool short_range_;
  bool long_range_;

  int target_;
  int neighbor_;

  std::string potential_output_;
  std::string force_output_;
  int force_print_width_;
  int force_print_precision_;

public:
  CoulombCommand(){
    // default value
    sigma_ = 2;
    cutoff_ = 10;
    nmax_ = 50;
    p_ = 6;
    quadrature_ = 5;
    level_ = 5;
    short_range_ = true;
    long_range_ = true;
    potential_output_ = "potential.dat";
    force_output_ = "force.dat";
    force_print_width_ = 25;
    force_print_precision_ = 15;

    target_ = 10;
    neighbor_ = 10;
  }

  void set_parameter(const std::map<std::string, std::string>& parameter){
    auto it = parameter.find("sigma");

    if(it != parameter.end()){
      sigma_ = std::stod(it->second);
    }

    it = parameter.find("cutoff");

    if(it != parameter.end()){
      cutoff_ = std::stod(it->second);
    }

    it = parameter.find("nmax");

    if(it != parameter.end()){
      nmax_ = std::stoi(it->second);
    }

    it = parameter.find("p");

    if(it != parameter.end()){
      p_ = std::stoi(it->second);
    }

    it = parameter.find("quadrature");

    if(it != parameter.end()){
      quadrature_ = std::stoi(it->second);
    }

    it = parameter.find("range");

    if(it != parameter.end()){
      if(it->second == "short_long"){
        short_range_ = true;
        long_range_ = true;
      }
      else if(it ->second == "short"){
        short_range_ = true;
        long_range_ = false;
      }
      else if(it ->second == "long"){
        short_range_ = false;
        long_range_ = true;
      }
      else{
        throw std::invalid_argument("range short_long|short|long");
      }
    }

    it = parameter.find("level");

    if(it != parameter.end()){
      level_ = std::stoi(it->second);
    }

    it = parameter.find("target");

    if(it != parameter.end()){
      target_ = std::stoi(it->second);
    }

    it = parameter.find("neighbor");

    if(it != parameter.end()){
      neighbor_ = std::stoi(it->second);
    }
  }

  void save_potential(real potential, std::ostream& o){
    std::ofstream out(potential_output_.c_str());

    if(!out){
      std::invalid_argument(potential_output_ + " can not be opened.");
    }

    Array::print_matrix(1, 1, &potential,
      force_print_width_, force_print_precision_, out);

    o << "output " << potential_output_ << std::endl;
  }

  void save_force(int natom, const real* force, std::ostream& o){
    std::ofstream out(force_output_.c_str());

    if(!out){
      std::invalid_argument(force_output_ + " can not be opened.");
    }

    Array::print_matrix(3 * natom, 1, force,
      force_print_width_, force_print_precision_, out);

    o << "output " << force_output_ << std::endl;
  }

  MSMLong* create_msm_long(int natom, const real* q){
    EwaldGaussianMixtureMSMKernel* kernel
      = new EwaldGaussianMixtureMSMKernel(sigma_, cutoff_, p_, quadrature_);

    return new MSMLong(kernel, p_, level_, natom, q);
  }


  void save_msm_potential_force(int natom, const real* q,
    const real* r, const real* L, std::ostream& o){
    real potential = 0;
    real* force = new real[3 * natom];

    if(short_range_){
      EwaldShort ewald_short(sigma_, cutoff_, natom, q);

      ewald_short.set_coordinate(r, L);
      ewald_short.print_parameter(o);
      potential += ewald_short.get_potential_force(force);
    }

    if(long_range_){
      Array::print_separator(o);
      TimeLogger log_long(false);

      log_long.start();

      real* long_force = new real[3 * natom];
      MSMLong* msm = create_msm_long(natom, q);

      msm->set_coordinate(r, L);
      msm->print_parameter(o);
      msm->set_grid_coefficient();
      potential += msm->get_potential_force(long_force);
      log_long.stop();
      log_long.print_time("total(msm_long)");

      for(int i = 0; i < 3 * natom; i++){
        force[i] += long_force[i];
      }

      delete msm;
    }

    Array::print_separator(o);
    save_potential(potential, o);
    save_force(natom, force, o);

    delete [] force;
  }

  void save_ewald_potential_force(int natom, const real* q,
    const real* r, const real* L, std::ostream& o){
    real potential = 0;
    real* force = new real[3 * natom];
    std::fill_n(force, 3 * natom, 0);

    if(short_range_){
      EwaldShort ewald_short(sigma_, cutoff_, natom, q);

      ewald_short.set_coordinate(r, L);
      ewald_short.print_parameter(o);
      potential += ewald_short.get_potential_force(force);
    }

    if(long_range_){
      Array::print_separator(o);
      EwaldLong ewald_long(sigma_, nmax_, natom, q);
      real* long_force = new real[3 * natom];

      ewald_long.set_coordinate(r, L);
      ewald_long.print_parameter(o);
      ewald_long.set_structure_factor();
      potential += ewald_long.get_potential_force(long_force);

      for(int i = 0; i < 3 * natom; i++){
        force[i] += long_force[i];
      }

      delete [] long_force;
    }

    Array::print_separator(o);
    save_potential(potential, o);
    save_force(natom, force, o);

    delete [] force;
  }

  void test_msm(int natom, const real* q, const real* r, const real* L,
    std::ostream& o){
    real* rtarget = new real[3 * target_];

    TestMSM::get_random_coordinate(L, target_, rtarget);
    o << "neighbor " << neighbor_ << std::endl;
    o << "target " << target_ << std::endl;
    o << "target_coordinate" << std::endl;
    Array::print_matrix(target_, 3, rtarget, o);

    MSMLong* msm = create_msm_long(natom, q);

    msm->set_coordinate(r, L);
    Array::print_separator(o);
    msm->print_parameter(o);

    Array::print_separator(o);
    EwaldLong ewald(sigma_, nmax_, natom, q);
    TestMSM::test_msm_long_periodic(msm, target_, rtarget, neighbor_,
      &ewald, o);

    delete msm;
    delete [] rtarget;
  }

};

#endif // COULOMB_COMMAND_H_
