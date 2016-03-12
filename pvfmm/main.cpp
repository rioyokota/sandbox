#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fftw3.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <omp.h>
#include <set>
#include <sstream>
#include <stack>
#include <stdint.h>
#include <string>
#include <sys/stat.h>
#include <vector>

#ifdef __SSE__
#include <xmmintrin.h>
#endif
#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __SSE3__
#include <pmmintrin.h>
#endif
#ifdef __AVX__
#include <immintrin.h>
#endif

//#include <vec.hpp>
#include <types.hpp>
#include <profile.hpp>
#include <mem_mgr.hpp>
#include <vector.hpp>
#include <matrix.hpp>
#include <precomp_mat.hpp>
#include <kernel.hpp>
#include <mortonid.hpp>
#include <sort.hpp>
#include <utils.hpp>
#include <fmm_node.hpp>
#include <interac_list.hpp>
#include <fmm_tree.hpp>

using namespace pvfmm;

int main(int argc, char **argv){
  omp_set_num_threads( atoi(commandline_option(argc, argv,  "-omp",     "1", false, "-omp  <int> =  (1)   : Number of OpenMP threads."          )));
  size_t N=  (size_t)strtod(commandline_option(argc, argv,    "-N",     "1",  true, "-N    <int>          : Number of points."                  ),NULL);
  size_t M=  (size_t)strtod(commandline_option(argc, argv,    "-M",   "350", false, "-M    <int>          : Number of points per octant."       ),NULL);
  int mult_order=   strtoul(commandline_option(argc, argv,    "-m",    "10", false, "-m    <int> = (10)   : Multipole order (+ve even integer)."),NULL,10);
  int depth=        strtoul(commandline_option(argc, argv,    "-d",    "15", false, "-d    <int> = (15)   : Maximum tree depth."                ),NULL,10);
  Profile::Enable(true);
  Profile::Tic("FMM_Test",true);
  Kernel potn_ker=BuildKernel<laplace_poten >("laplace"    , std::pair<int,int>(1,1));
  Kernel grad_ker=BuildKernel<laplace_grad >("laplace_grad", std::pair<int,int>(1,3),
					     &potn_ker, &potn_ker, NULL, &potn_ker, &potn_ker, NULL, &potn_ker, NULL);
  typename FMM_Node::NodeData tree_data;
  tree_data.max_depth=depth;
  tree_data.max_pts=M;
  std::vector<Real_t> src_coord, src_value;
  srand48(0);
  for(size_t i=0;i<N;i++){
    src_coord.push_back(drand48());
    src_coord.push_back(drand48());
    src_coord.push_back(drand48());
  }
  for(size_t i=0;i<src_coord.size()/3;i++) src_value.push_back(drand48()-0.5);
  tree_data.coord=src_coord;
  tree_data.value=src_value;
  FMM_Tree tree;
  tree.Initialize(mult_order,&grad_ker);
  Vector<Real_t> trg_value;
  for(size_t it=0;it<2;it++){
    Profile::Tic("TotalTime",true);
    tree.Initialize(&tree_data);
    Profile::Tic("SetSrcTrg",true);
    std::vector<FMM_Node*>& node=tree.GetNodeList();
#pragma omp parallel for
    for(size_t i=0;i<node.size();i++){
      node[i]->  trg_coord.ReInit(node[i]->  pt_coord.Dim(), &node[i]->  pt_coord[0]);
      node[i]->  src_coord.ReInit(node[i]->  pt_coord.Dim(), &node[i]->  pt_coord[0]);
      node[i]->  src_value.ReInit(node[i]->  pt_value.Dim(), &node[i]->  pt_value[0]);
      node[i]->trg_scatter.ReInit(node[i]->pt_scatter.Dim(), &node[i]->pt_scatter[0]);
      node[i]->src_scatter.ReInit(node[i]->pt_scatter.Dim(), &node[i]->pt_scatter[0]);
    }
    Profile::Toc();
    tree.InitFMM_Tree(false);
    tree.SetupFMM();
    tree.RunFMM();
    Profile::Toc();
  }
  long nleaf=0, maxdepth=0;
  std::vector<size_t> all_nodes(MAX_DEPTH+1,0);
  std::vector<size_t> leaf_nodes(MAX_DEPTH+1,0);
  std::vector<FMM_Node*>& nodes=tree.GetNodeList();
  for(size_t i=0;i<nodes.size();i++){
    FMM_Node* n=nodes[i];
    if(!n->IsGhost()) all_nodes[n->depth]++;
    if(!n->IsGhost() && n->IsLeaf()){
      leaf_nodes[n->depth]++;
      if(maxdepth<n->depth) maxdepth=n->depth;
      nleaf++;
    }
  }
  std::cout<<"Leaf Nodes : "<<nleaf<<'\n';
  std::cout<<"Tree Depth : "<<maxdepth<<'\n';
  tree.CheckFMMOutput("Output");
  Profile::Toc();
  Profile::print();
  return 0;
}

