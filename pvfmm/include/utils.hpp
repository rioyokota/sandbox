#ifndef _UTILS_
#define _UTILS_

#include <vector>
#include <cheb_utils.hpp>
#include <fmm_tree.hpp>

template<typename FMMNode>
void CheckFMMOutput(pvfmm::FMM_Tree<FMMNode>* mytree, const pvfmm::Kernel<Real_t>* mykernel);

template <class Real_t>
struct TestFn{
  typedef void (*Fn_t)(Real_t* c, int n, Real_t* out);
};

enum DistribType{
  UnifGrid,
  RandUnif,
  RandGaus,
  RandElps,
  RandSphr
};

void commandline_option_start(int argc, char** argv, const char* help_text=NULL);

const char* commandline_option(int argc, char** argv, const char* opt, const char* def_val, bool required, const char* err_msg);

void commandline_option_end(int argc, char** argv);

#include <utils.txx>

#endif
