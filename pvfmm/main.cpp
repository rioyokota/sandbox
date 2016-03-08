#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fftw3.h>
#include <fstream>
#include <iomanip>
#include <iostream>
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

#include <types.hpp>
#include <profile.hpp>
#include <vector.hpp>
#include <matrix.hpp>
#include <precomp_mat.hpp>
#include <intrin_wrapper.hpp>
#include <kernel.hpp>
#include <fft_wrapper.hpp>
#include <mortonid.hpp>
#include <sort.hpp>
#include <interac_list.hpp>
#include <utils.hpp>
#include <fmm_node.hpp>
#include <fmm_tree.hpp>

int main(int argc, char **argv){
  commandline_option_start(argc, argv);
  omp_set_num_threads( atoi(commandline_option(argc, argv,  "-omp",     "1", false, "-omp  <int> =  (1)   : Number of OpenMP threads."          )));
  size_t N=  (size_t)strtod(commandline_option(argc, argv,    "-N",     "1",  true, "-N    <int>          : Number of points."                  ),NULL);
  size_t M=  (size_t)strtod(commandline_option(argc, argv,    "-M",   "350", false, "-M    <int>          : Number of points per octant."       ),NULL);
  int mult_order=   strtoul(commandline_option(argc, argv,    "-m",    "10", false, "-m    <int> = (10)   : Multipole order (+ve even integer)."),NULL,10);
  int depth=        strtoul(commandline_option(argc, argv,    "-d",    "15", false, "-d    <int> = (15)   : Maximum tree depth."                ),NULL,10);
  commandline_option_end(argc, argv);
  pvfmm::Profile::Enable(true);
  pvfmm::Profile::Tic("FMM_Test",true);
  typedef pvfmm::FMM_Node FMMNode_t;
  typedef pvfmm::FMM_Tree FMMTree_t;
  const pvfmm::Kernel<Real_t>* mykernel = &pvfmm::LaplaceKernel<Real_t>::gradient();
  typename FMMNode_t::NodeData tree_data;
  tree_data.max_depth=depth;
  tree_data.max_pts=M;
  std::vector<Real_t> src_coord, src_value;
  src_coord=point_distrib(N);
  for(size_t i=0;i<src_coord.size()/3;i++) src_value.push_back(drand48()-0.5);
  tree_data.coord=src_coord;
  tree_data.value=src_value;
  FMMTree_t tree;
  tree.Initialize(mult_order,mykernel);
  pvfmm::Vector<Real_t> trg_value;
  for(size_t it=0;it<2;it++){
    pvfmm::Profile::Tic("TotalTime",true);
    tree.Initialize(&tree_data);
    pvfmm::Profile::Tic("SetSrcTrg",true);
    std::vector<FMMNode_t*>& node=tree.GetNodeList();
#pragma omp parallel for
    for(size_t i=0;i<node.size();i++){
      node[i]->  trg_coord.ReInit(node[i]->  pt_coord.Dim(), &node[i]->  pt_coord[0]);
      node[i]->  src_coord.ReInit(node[i]->  pt_coord.Dim(), &node[i]->  pt_coord[0]);
      node[i]->  src_value.ReInit(node[i]->  pt_value.Dim(), &node[i]->  pt_value[0]);
      node[i]->trg_scatter.ReInit(node[i]->pt_scatter.Dim(), &node[i]->pt_scatter[0]);
      node[i]->src_scatter.ReInit(node[i]->pt_scatter.Dim(), &node[i]->pt_scatter[0]);
    }
    pvfmm::Profile::Toc();
    tree.InitFMM_Tree(false);
    tree.SetupFMM();
    tree.RunFMM();
    pvfmm::Profile::Toc();
  }
  long nleaf=0, maxdepth=0;
  std::vector<size_t> all_nodes(MAX_DEPTH+1,0);
  std::vector<size_t> leaf_nodes(MAX_DEPTH+1,0);
  std::vector<FMMNode_t*>& nodes=tree.GetNodeList();
  for(size_t i=0;i<nodes.size();i++){
    FMMNode_t* n=nodes[i];
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
  pvfmm::Profile::Toc();
  pvfmm::Profile::print();
  return 0;
}

