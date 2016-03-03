#include <omp.h>
#include <iostream>
#include <string>
#include <vector>
#include <stack>

#include <pvfmm_common.hpp>

#ifndef _PVFMM_PROFILE_HPP_
#define _PVFMM_PROFILE_HPP_

namespace pvfmm{

class Profile{
public:
  
  static long long Add_FLOP(long long inc){
    long long orig_val=FLOP;
#pragma omp atomic update
    FLOP+=inc;
    return orig_val;
  }

  static long long Add_MEM(long long inc){
    long long orig_val=MEM;
#pragma omp atomic update
    MEM+=inc;
    for(size_t i=0;i<max_mem.size();i++){
      if(max_mem[i]<MEM) max_mem[i]=MEM;
    }
    return orig_val;
  }

  static bool Enable(bool state){
    bool orig_val=enable_state;
    enable_state=state;
    return orig_val;
  }

  static void Tic(const char* name_, bool sync_=false, int verbose=0){
    if(!enable_state) return;
    if(verbose<=5 && verb_level.size()==enable_depth){
      name.push(name_);
      sync.push(sync_);
      max_mem.push_back(MEM);
      e_log.push_back(true);
      s_log.push_back(sync_);
      n_log.push_back(name.top());
      t_log.push_back(omp_get_wtime());
      f_log.push_back(FLOP);
      m_log.push_back(MEM);
      max_m_log.push_back(MEM);
      enable_depth++;
    }
    verb_level.push(verbose);
  }


    static void Toc();

    static void print();

    static void reset();
  private:

  static long long FLOP;
  static long long MEM;
  static bool enable_state;
  static std::stack<bool> sync;
  static std::stack<std::string> name;
  static std::vector<long long> max_mem;

  static unsigned int enable_depth;
  static std::stack<int> verb_level;

  static std::vector<bool> e_log;
  static std::vector<bool> s_log;
  static std::vector<std::string> n_log;
  static std::vector<double> t_log;
  static std::vector<long long> f_log;
  static std::vector<long long> m_log;
  static std::vector<long long> max_m_log;
};

}//end namespace

#endif //_PVFMM_PROFILE_HPP_
