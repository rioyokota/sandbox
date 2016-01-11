#include <pvfmm_common.hpp>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <stdio.h>

#include <profile.hpp>
#include <fmm_pts.hpp>
#include <fmm_node.hpp>
#include <fmm_tree.hpp>
#include <utils.hpp>

template <class Real_t>
void fmm_test(size_t N, size_t M, Real_t b, int dist, int mult_order, int depth){
  typedef pvfmm::FMM_Node<pvfmm::MPI_Node<Real_t> > FMMNode_t;
  typedef pvfmm::FMM_Pts<FMMNode_t> FMM_Mat_t;
  typedef pvfmm::FMM_Tree<FMM_Mat_t> FMM_Tree_t;

  //Set kernel.
  pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
  const pvfmm::Kernel<Real_t>* mykernel = &pvfmm::LaplaceKernel<Real_t>::gradient();

  // Find out number of OMP thereads.
  int omp_p=omp_get_max_threads();

  // Find out my identity in the default communicator
  int myrank=0, p=1;

  //Various parameters.
  typename FMMNode_t::NodeData tree_data;
  tree_data.dim=COORD_DIM;
  tree_data.max_depth=depth;
  tree_data.max_pts=M; // Points per octant.

  { //Set particle coordinates and values.
    std::vector<Real_t> src_coord, src_value;
    src_coord=point_distrib<Real_t>((dist==0?UnifGrid:(dist==1?RandSphr:RandElps)),N);
    for(size_t i=0;i<src_coord.size();i++) src_coord[i]*=b;
    for(size_t i=0;i<src_coord.size()*mykernel->ker_dim[0]/COORD_DIM;i++) src_value.push_back(drand48()-0.5);
    tree_data.pt_coord=src_coord;
    tree_data.pt_value=src_value;
  }

  //Initialize FMM_Mat.
  FMM_Mat_t fmm_mat;
  fmm_mat.Initialize(mult_order,mykernel);

  //Create Tree.
  FMM_Tree_t tree;

  pvfmm::Vector<Real_t> trg_value;
  for(size_t i=0;i<2;i++){ // Compute potential
    pvfmm::Profile::Tic("TotalTime",true);

    //Initialize tree with input data.
    tree.Initialize(&tree_data);

    //Initialize FMM Tree
    pvfmm::Profile::Tic("SetSrcTrg",true);
    { // Set src and trg points
      std::vector<FMMNode_t*>& node=tree.GetNodeList();
      #pragma omp parallel for
      for(size_t i=0;i<node.size();i++){
        node[i]->  trg_coord.ReInit(node[i]->  pt_coord.Dim(), &node[i]->  pt_coord[0]);
        node[i]->  src_coord.ReInit(node[i]->  pt_coord.Dim(), &node[i]->  pt_coord[0]);
        node[i]->  src_value.ReInit(node[i]->  pt_value.Dim(), &node[i]->  pt_value[0]);
        node[i]->trg_scatter.ReInit(node[i]->pt_scatter.Dim(), &node[i]->pt_scatter[0]);
        node[i]->src_scatter.ReInit(node[i]->pt_scatter.Dim(), &node[i]->pt_scatter[0]);
      }
    }
    pvfmm::Profile::Toc();
    tree.InitFMM_Tree(false,bndry);

    // Setup FMM
    tree.SetupFMM(&fmm_mat);
    tree.RunFMM();

    { // Scatter trg values
      pvfmm::Profile::Tic("Scatter",true);
      pvfmm::Profile::Toc();
    }
    pvfmm::Profile::Toc();
  }

  { //Output max tree depth.
    long nleaf=0, maxdepth=0;
    std::vector<size_t> all_nodes(MAX_DEPTH+1,0);
    std::vector<size_t> leaf_nodes(MAX_DEPTH+1,0);
    std::vector<FMMNode_t*>& nodes=tree.GetNodeList();
    for(size_t i=0;i<nodes.size();i++){
      FMMNode_t* n=nodes[i];
      if(!n->IsGhost()) all_nodes[n->Depth()]++;
      if(!n->IsGhost() && n->IsLeaf()){
        leaf_nodes[n->Depth()]++;
        if(maxdepth<n->Depth()) maxdepth=n->Depth();
        nleaf++;
      }
    }
    if(!myrank) std::cout<<"Leaf Nodes : "<<nleaf<<'\n';
    if(!myrank) std::cout<<"Tree Depth : "<<maxdepth<<'\n';
  }

  //Find error in FMM output.
  CheckFMMOutput<FMM_Mat_t>(&tree, mykernel, "Output");
}

int main(int argc, char **argv){
  // Read command line options.
  commandline_option_start(argc, argv);
  omp_set_num_threads( atoi(commandline_option(argc, argv,  "-omp",     "1", false, "-omp  <int> =  (1)   : Number of OpenMP threads."          )));
  size_t   N=(size_t)strtod(commandline_option(argc, argv,    "-N",     "1",  true, "-N    <int>          : Number of points."                  ),NULL);
  size_t   M=(size_t)strtod(commandline_option(argc, argv,    "-M",   "350", false, "-M    <int>          : Number of points per octant."       ),NULL);
  double   b=        strtod(commandline_option(argc, argv,    "-b",     "1", false, "-b    <int> =  (1)   : Bounding-box length (0 < b <= 1)"   ),NULL);
  int      m=       strtoul(commandline_option(argc, argv,    "-m",    "10", false, "-m    <int> = (10)   : Multipole order (+ve even integer)."),NULL,10);
  int      d=       strtoul(commandline_option(argc, argv,    "-d",    "15", false, "-d    <int> = (15)   : Maximum tree depth."                ),NULL,10);
  bool    sp=   (1==strtoul(commandline_option(argc, argv,   "-sp",     "0", false, "-sp   <0/1> =  (0)   : Single Precision."                  ),NULL,10));
  int   dist=       strtoul(commandline_option(argc, argv, "-dist",     "0", false, "-dist <int> =  (0)   : 0) Unif 1) Sphere 2) Ellipse"       ),NULL,10);
  commandline_option_end(argc, argv);
  pvfmm::Profile::Enable(true);

  // Run FMM with above options.
  pvfmm::Profile::Tic("FMM_Test",true);
  if(sp) fmm_test<float >(N, M, b, dist, m, d);
  else   fmm_test<double>(N, M, b, dist, m, d);
  pvfmm::Profile::Toc();

  //Output Profiling results.
  pvfmm::Profile::print();

  // Shut down MPI
  return 0;
}

