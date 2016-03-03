#include <omp.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cassert>
#include <cstdlib>
#include <string>
#include <vector>
#include <stack>
#include <pvfmm_common.hpp>
#include <profile.hpp>

namespace pvfmm{

long long Profile::Add_FLOP(long long inc){
  long long orig_val=FLOP;
  #pragma omp atomic update
  FLOP+=inc;
  return orig_val;
}

long long Profile::Add_MEM(long long inc){
  long long orig_val=MEM;
  #pragma omp atomic update
  MEM+=inc;
  for(size_t i=0;i<max_mem.size();i++){
    if(max_mem[i]<MEM) max_mem[i]=MEM;
  }
  return orig_val;
}

bool Profile::Enable(bool state){
  bool orig_val=enable_state;
  enable_state=state;
  return orig_val;
}

void Profile::Tic(const char* name_, bool sync_, int verbose){
  if(!enable_state) return;
  if(verbose<=__PROFILE__ && verb_level.size()==enable_depth){
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

void Profile::Toc(){
  if(!enable_state) return;
  if(verb_level.top()<=__PROFILE__ && verb_level.size()==enable_depth){
    std::string name_=name.top();
    bool sync_=sync.top();
    e_log.push_back(false);
    s_log.push_back(sync_);
    n_log.push_back(name_);
    t_log.push_back(omp_get_wtime());
    f_log.push_back(FLOP);
    m_log.push_back(MEM);
    max_m_log.push_back(max_mem.back());
    name.pop();
    sync.pop();
    max_mem.pop_back();
    enable_depth--;
  }
  verb_level.pop();
}

void Profile::print(){
  int np=1, rank=0;
  std::stack<double> tt;
  std::stack<long long> ff;
  std::stack<long long> mm;
  int width=10;
  size_t level=0;

  std::stack<std::string> out_stack;
  std::string s;
  out_stack.push(s);
  for(size_t i=0;i<e_log.size();i++){
    if(e_log[i]){
      level++;
      tt.push(t_log[i]);
      ff.push(f_log[i]);
      mm.push(m_log[i]);

      std::string ss;
      out_stack.push(ss);
    }else{
      double t0=t_log[i]-tt.top();tt.pop();
      double f0=(double)(f_log[i]-ff.top())*1e-9;ff.pop();
      double fs0=f0/t0;
      double t_max=t0, t_min=t0, t_sum=t0, t_avg=t0;
      double f_max=f0, f_min=f0, f_sum=f0, f_avg=f0;
      double fs_max=fs0, fs_min=fs0, fs_sum=fs0;//, fs_avg;
      double m_init, m_max, m_final;

      m_final=(double)m_log[i]*1e-9;
      m_init =(double)mm.top()*1e-9; mm.pop();
      m_max  =(double)max_m_log[i]*1e-9;

      t_avg=t_sum/np;
      f_avg=f_sum/np;
      fs_sum=f_sum/t_max;
 
      if(!rank){
	if(i==151||i==153)
	  std::cout << n_log[i] << "     : " << t_avg << std::endl;
      }
      level--;
    }
  }
  if(!rank)
    std::cout<<out_stack.top()<<'\n';

  reset();
}

void Profile::reset(){
  FLOP=0;
  while(!sync.empty())sync.pop();
  while(!name.empty())name.pop();

  e_log.clear();
  s_log.clear();
  n_log.clear();
  t_log.clear();
  f_log.clear();
  m_log.clear();
  max_m_log.clear();
}

long long Profile::FLOP=0;
long long Profile::MEM=0;
bool Profile::enable_state=false;
std::stack<bool> Profile::sync;
std::stack<std::string> Profile::name;
std::vector<long long> Profile::max_mem;

unsigned int Profile::enable_depth=0;
std::stack<int> Profile::verb_level;

std::vector<bool> Profile::e_log;
std::vector<bool> Profile::s_log;
std::vector<std::string> Profile::n_log;
std::vector<double> Profile::t_log;
std::vector<long long> Profile::f_log;
std::vector<long long> Profile::m_log;
std::vector<long long> Profile::max_m_log;

}//end namespace
