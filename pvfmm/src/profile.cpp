/**
 * \file profile.cpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 2-11-2011
 * \brief This file contains implementation of the class Profile.
 */

#include <mpi.h>
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
  #if __PROFILE__ >= 0
  #pragma omp atomic update
  FLOP+=inc;
  #endif
  return orig_val;
}

long long Profile::Add_MEM(long long inc){
  long long orig_val=MEM;
  #if __PROFILE__ >= 0
  #pragma omp atomic update
  MEM+=inc;
  for(size_t i=0;i<max_mem.size();i++){
    if(max_mem[i]<MEM) max_mem[i]=MEM;
  }
  #endif
  return orig_val;
}

bool Profile::Enable(bool state){
  bool orig_val=enable_state;
  #if __PROFILE__ >= 0
  enable_state=state;
  #endif
  return orig_val;
}

void Profile::Tic(const char* name_, const MPI_Comm* comm_,bool sync_, int verbose){
#if __PROFILE__ >= 0
  if(!enable_state) return;
  if(verbose<=__PROFILE__ && verb_level.size()==enable_depth){
    #ifdef __VERBOSE__
    int rank=0;
    if(!rank){
      for(size_t i=0;i<name.size();i++) std::cout<<"    ";
      std::cout << "\033[1;31m"<<name_<<"\033[0m {\n";
    }
    #endif
    name.push(name_);
    comm.push((MPI_Comm*)comm_);
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
#endif
}

void Profile::Toc(){
#if __PROFILE__ >= 0
  if(!enable_state) return;
  if(verb_level.top()<=__PROFILE__ && verb_level.size()==enable_depth){
    std::string name_=name.top();
    MPI_Comm* comm_=comm.top();
    bool sync_=sync.top();
    //sync_=true;

    e_log.push_back(false);
    s_log.push_back(sync_);
    n_log.push_back(name_);
    t_log.push_back(omp_get_wtime());
    f_log.push_back(FLOP);

    m_log.push_back(MEM);
    max_m_log.push_back(max_mem.back());

    name.pop();
    comm.pop();
    sync.pop();
    max_mem.pop_back();

    #ifdef __VERBOSE__
    int rank=0;
    if(!rank){
      for(size_t i=0;i<name.size();i++) std::cout<<"    ";
      std::cout<<"}\n";
    }
    #endif
    enable_depth--;
  }
  verb_level.pop();
#endif
}

void Profile::print(const MPI_Comm* comm_){
#if __PROFILE__ >= 0
  int np=1, rank=0;
  MPI_Comm c_self=MPI_COMM_SELF;
  if(comm_==NULL) comm_=&c_self;

  std::stack<double> tt;
  std::stack<long long> ff;
  std::stack<long long> mm;
  int width=10;
  size_t level=0;
#ifdef __VERBOSE__
  if(!rank && e_log.size()>0){
    std::cout<<"\n"<<std::setw(width*3-2*level)<<" ";
    std::cout<<"  "<<std::setw(width)<<"t_min";
    std::cout<<"  "<<std::setw(width)<<"t_avg";
    std::cout<<"  "<<std::setw(width)<<"t_max";
    std::cout<<"  "<<std::setw(width)<<"f_min";
    std::cout<<"  "<<std::setw(width)<<"f_avg";
    std::cout<<"  "<<std::setw(width)<<"f_max";

    std::cout<<"  "<<std::setw(width)<<"f/s_min";
    std::cout<<"  "<<std::setw(width)<<"f/s_max";
    std::cout<<"  "<<std::setw(width)<<"f/s_total";

    std::cout<<"  "<<std::setw(width)<<"m_init";
    std::cout<<"  "<<std::setw(width)<<"m_max";
    std::cout<<"  "<<std::setw(width)<<"m_final"<<'\n';
  }
#endif

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
      //fs_avg=f_avg/t_max;
      fs_sum=f_sum/t_max;
 
      if(!rank){
#ifdef __VERBOSE__
	std::string s0=out_stack.top();out_stack.pop();
        std::string s1=out_stack.top();out_stack.pop();
        std::stringstream ss(std::stringstream::in | std::stringstream::out);
        ss<<setiosflags(std::ios::fixed)<<std::setprecision(4)<<std::setiosflags(std::ios::left);

        for(size_t j=0;j<level-1;j++){
          size_t l=i+1;
          size_t k=level-1;
          while(k>j && l<e_log.size()){
            k+=(e_log[l]?1:-1);
            l++;
          }
          if(l<e_log.size()?e_log[l]:false)
            ss<<"| ";
          else
            ss<<"  ";
        }
        ss<<"+-";
        ss<<std::setw(width*3-2*level)<<n_log[i];
        ss<<std::setiosflags(std::ios::right);
        ss<<"  "<<std::setw(width)<<t_min;
        ss<<"  "<<std::setw(width)<<t_avg;
        ss<<"  "<<std::setw(width)<<t_max;

        ss<<"  "<<std::setw(width)<<f_min;
        ss<<"  "<<std::setw(width)<<f_avg;
        ss<<"  "<<std::setw(width)<<f_max;

        ss<<"  "<<std::setw(width)<<fs_min;
        //ss<<"  "<<std::setw(width)<<fs_avg;
        ss<<"  "<<std::setw(width)<<fs_max;
        ss<<"  "<<std::setw(width)<<fs_sum;

        ss<<"  "<<std::setw(width)<<m_init;
        ss<<"  "<<std::setw(width)<<m_max;
        ss<<"  "<<std::setw(width)<<m_final<<'\n';

        s1+=ss.str()+s0;
        if(!s0.empty() && (i+1<e_log.size()?e_log[i+1]:false)){
          for(size_t j=0;j<level;j++){
            size_t l=i+1;
            size_t k=level-1;
            while(k>j && l<e_log.size()){
              k+=(e_log[l]?1:-1);
              l++;
            }
            if(l<e_log.size()?e_log[l]:false) s1+="| ";
            else s1+="  ";
          }
          s1+="\n";
        }// */
        out_stack.push(s1);
#else
	if(i==167||i==169) std::cout << n_log[i] << "     : " << setiosflags(std::ios::fixed) << std::setprecision(4) << t_avg << std::endl;
#endif
      }
      level--;
    }
  }
  if(!rank)
    std::cout<<out_stack.top()<<'\n';

  reset();
#endif
}

void Profile::reset(){
  FLOP=0;
  while(!sync.empty())sync.pop();
  while(!name.empty())name.pop();
  while(!comm.empty())comm.pop();

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
std::stack<MPI_Comm*> Profile::comm;
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
