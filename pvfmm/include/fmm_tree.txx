/**
 * \file fmm_tree.txx
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 12-11-2010
 * \brief This file contains the implementation of the class FMM_Tree.
 */

#include <omp.h>
#include <sstream>
#include <iomanip>
#include <cassert>
#include <cstdlib>

#include <mpi_node.hpp>
#include <fmm_node.hpp>
#include <mem_mgr.hpp>
#include <mortonid.hpp>
#include <profile.hpp>
#include <vector.hpp>

namespace pvfmm{

template <class FMM_Mat_t>
void FMM_Tree<FMM_Mat_t>::Initialize(typename Node_t::NodeData* init_data) {
  Profile::Tic("InitTree",true);{

  //Build octree from points.
  MPI_Tree<Node_t>::Initialize(init_data);

  Profile::Tic("InitFMMData",true,5);
  { //Initialize FMM data.
    std::vector<Node_t*>& nodes=this->GetNodeList();
    #pragma omp parallel for
    for(size_t i=0;i<nodes.size();i++){
      if(nodes[i]->FMMData()==NULL) nodes[i]->FMMData()=mem::aligned_new<typename FMM_Mat_t::FMMData>();
    }
  }
  Profile::Toc();

  }Profile::Toc();
}


template <class FMM_Mat_t>
void FMM_Tree<FMM_Mat_t>::InitFMM_Tree(bool refine, BoundaryType bndry_) {
  Profile::Tic("InitFMM_Tree",true);{

  interac_list.Initialize(this->Dim());
  bndry=bndry_;

  if(refine){
    //RefineTree
    Profile::Tic("RefineTree",true,5);
    this->RefineTree();
    Profile::Toc();
  }

  //2:1 Balancing
  Profile::Tic("2:1Balance",true,5);
  this->Balance21(bndry);
  Profile::Toc();

  //Redistribute nodes.
//  Profile::Tic("Redistribute",true,5);
//  this->RedistNodes();
//  Profile::Toc();

  }Profile::Toc();
}


template <class FMM_Mat_t>
void FMM_Tree<FMM_Mat_t>::SetupFMM(FMM_Mat_t* fmm_mat_) {
  Profile::Tic("SetupFMM",true);{
  typedef typename FMM_Mat_t::FMMTree_t MatTree_t;

  //int omp_p=omp_get_max_threads();
  if(fmm_mat!=fmm_mat_){ // Clear previous setup
    setup_data.clear();
    precomp_lst.clear();
    fmm_mat=fmm_mat_;
  }

  //Construct LET
  Profile::Tic("ConstructLET",false,2);
  this->ConstructLET(bndry);
  Profile::Toc();

  //Set Colleagues (Needed to build U, V, W and X lists.)
  Profile::Tic("SetColleagues",false,3);
  this->SetColleagues(bndry);
  Profile::Toc();

  Profile::Tic("CollectNodeData",false,3);
  //Build node list.
  Node_t* n=dynamic_cast<Node_t*>(this->PostorderFirst());
  std::vector<Node_t*> all_nodes;
  while(n!=NULL){
    n->pt_cnt[0]=0;
    n->pt_cnt[1]=0;
    all_nodes.push_back(n);
    n=static_cast<Node_t*>(this->PostorderNxt(n));
  }
  //Collect node data into continuous array.
  std::vector<Vector<Node_t*> > node_lists; // TODO: Remove this parameter, not really needed
  fmm_mat->CollectNodeData((MatTree_t*)this,all_nodes, node_data_buff, node_lists);
  Profile::Toc();

  Profile::Tic("BuildLists",false,3);
  BuildInteracLists();
  Profile::Toc();

  setup_data.resize(8*MAX_DEPTH);
  precomp_lst.resize(8);

  Profile::Tic("UListSetup",false,3);
  for(size_t i=0;i<MAX_DEPTH;i++){
    setup_data[i+MAX_DEPTH*0].precomp_data=&precomp_lst[0];
    fmm_mat->U_ListSetup(setup_data[i+MAX_DEPTH*0],(MatTree_t*)this,node_data_buff,node_lists,fmm_mat->ScaleInvar()?(i==0?-1:MAX_DEPTH+1):i);
  }
  Profile::Toc();
  Profile::Tic("WListSetup",false,3);
  for(size_t i=0;i<MAX_DEPTH;i++){
    setup_data[i+MAX_DEPTH*1].precomp_data=&precomp_lst[1];
    fmm_mat->W_ListSetup(setup_data[i+MAX_DEPTH*1],(MatTree_t*)this,node_data_buff,node_lists,fmm_mat->ScaleInvar()?(i==0?-1:MAX_DEPTH+1):i);
  }
  Profile::Toc();
  Profile::Tic("XListSetup",false,3);
  for(size_t i=0;i<MAX_DEPTH;i++){
    setup_data[i+MAX_DEPTH*2].precomp_data=&precomp_lst[2];
    fmm_mat->X_ListSetup(setup_data[i+MAX_DEPTH*2],(MatTree_t*)this,node_data_buff,node_lists,fmm_mat->ScaleInvar()?(i==0?-1:MAX_DEPTH+1):i);
  }
  Profile::Toc();
  Profile::Tic("VListSetup",false,3);
  for(size_t i=0;i<MAX_DEPTH;i++){
    setup_data[i+MAX_DEPTH*3].precomp_data=&precomp_lst[3];
    fmm_mat->V_ListSetup(setup_data[i+MAX_DEPTH*3],(MatTree_t*)this,node_data_buff,node_lists,fmm_mat->ScaleInvar()?(i==0?-1:MAX_DEPTH+1):i);
  }
  Profile::Toc();
  Profile::Tic("D2DSetup",false,3);
  for(size_t i=0;i<MAX_DEPTH;i++){
    setup_data[i+MAX_DEPTH*4].precomp_data=&precomp_lst[4];
    fmm_mat->Down2DownSetup(setup_data[i+MAX_DEPTH*4],(MatTree_t*)this,node_data_buff,node_lists,i);
  }
  Profile::Toc();
  Profile::Tic("D2TSetup",false,3);
  for(size_t i=0;i<MAX_DEPTH;i++){
    setup_data[i+MAX_DEPTH*5].precomp_data=&precomp_lst[5];
    fmm_mat->Down2TargetSetup(setup_data[i+MAX_DEPTH*5],(MatTree_t*)this,node_data_buff,node_lists,fmm_mat->ScaleInvar()?(i==0?-1:MAX_DEPTH+1):i);
  }
  Profile::Toc();

  Profile::Tic("S2USetup",false,3);
  for(size_t i=0;i<MAX_DEPTH;i++){
    setup_data[i+MAX_DEPTH*6].precomp_data=&precomp_lst[6];
    fmm_mat->Source2UpSetup(setup_data[i+MAX_DEPTH*6],(MatTree_t*)this,node_data_buff,node_lists,fmm_mat->ScaleInvar()?(i==0?-1:MAX_DEPTH+1):i);
  }
  Profile::Toc();
  Profile::Tic("U2USetup",false,3);
  for(size_t i=0;i<MAX_DEPTH;i++){
    setup_data[i+MAX_DEPTH*7].precomp_data=&precomp_lst[7];
    fmm_mat->Up2UpSetup(setup_data[i+MAX_DEPTH*7],(MatTree_t*)this,node_data_buff,node_lists,i);
  }
  Profile::Toc();

  ClearFMMData();

  }Profile::Toc();
}

template <class FMM_Mat_t>
void FMM_Tree<FMM_Mat_t>::ClearFMMData() {
  Profile::Tic("ClearFMMData",true);{

  int omp_p=omp_get_max_threads();
  #pragma omp parallel for
  for(int j=0;j<omp_p;j++){
    Matrix<Real_t>* mat;

    mat=setup_data[0+MAX_DEPTH*1]. input_data;
    if(mat && mat->Dim(0)*mat->Dim(1)){
      size_t a=(mat->Dim(0)*mat->Dim(1)*(j+0))/omp_p;
      size_t b=(mat->Dim(0)*mat->Dim(1)*(j+1))/omp_p;
      memset(&(*mat)[0][a],0,(b-a)*sizeof(Real_t));
    }

    mat=setup_data[0+MAX_DEPTH*2].output_data;
    if(mat && mat->Dim(0)*mat->Dim(1)){
      size_t a=(mat->Dim(0)*mat->Dim(1)*(j+0))/omp_p;
      size_t b=(mat->Dim(0)*mat->Dim(1)*(j+1))/omp_p;
      memset(&(*mat)[0][a],0,(b-a)*sizeof(Real_t));
    }

    mat=setup_data[0+MAX_DEPTH*0].output_data;
    if(mat && mat->Dim(0)*mat->Dim(1)){
      size_t a=(mat->Dim(0)*mat->Dim(1)*(j+0))/omp_p;
      size_t b=(mat->Dim(0)*mat->Dim(1)*(j+1))/omp_p;
      memset(&(*mat)[0][a],0,(b-a)*sizeof(Real_t));
    }
  }

  }Profile::Toc();
}


template <class FMM_Mat_t>
void FMM_Tree<FMM_Mat_t>::RunFMM() {
  Profile::Tic("RunFMM",true);{

  //Upward Pass
  Profile::Tic("UpwardPass",false,2);
  UpwardPass();
  Profile::Toc();

  //Multipole Reduce Broadcast.
  Profile::Tic("ReduceBcast",true,2);
  //  MultipoleReduceBcast();
  Profile::Toc();

  //Downward Pass
  Profile::Tic("DownwardPass",true,2);
  DownwardPass();
  Profile::Toc();

  }Profile::Toc();
}


template <class FMM_Mat_t>
void FMM_Tree<FMM_Mat_t>::UpwardPass() {

  int max_depth=0;
  { // Get max_depth
    int max_depth_loc=0;
    std::vector<Node_t*>& nodes=this->GetNodeList();
    for(size_t i=0;i<nodes.size();i++){
      Node_t* n=nodes[i];
      if(n->Depth()>max_depth_loc) max_depth_loc=n->Depth();
    }
    max_depth = max_depth_loc;
  }

  //Upward Pass (initialize all leaf nodes)
  Profile::Tic("S2U",false,5);
  for(int i=0; i<=(fmm_mat->ScaleInvar()?0:max_depth); i++){ // Source2Up
    if(!fmm_mat->ScaleInvar()) fmm_mat->SetupPrecomp(setup_data[i+MAX_DEPTH*6]);
    fmm_mat->Source2Up(setup_data[i+MAX_DEPTH*6]);
  }
  Profile::Toc();

  //Upward Pass (level by level)
  Profile::Tic("U2U",false,5);
  for(int i=max_depth-1; i>=0; i--){ // Up2Up
    if(!fmm_mat->ScaleInvar()) fmm_mat->SetupPrecomp(setup_data[i+MAX_DEPTH*7]);
    fmm_mat->Up2Up(setup_data[i+MAX_DEPTH*7]);
  }
  Profile::Toc();
}


template <class FMM_Mat_t>
void FMM_Tree<FMM_Mat_t>::BuildInteracLists() {
  std::vector<Node_t*> n_list_src;
  std::vector<Node_t*> n_list_trg;
  { // Build n_list
    std::vector<Node_t*>& nodes=this->GetNodeList();
    for(size_t i=0;i<nodes.size();i++){
      if(!nodes[i]->IsGhost() && nodes[i]->pt_cnt[0]){
        n_list_src.push_back(nodes[i]);
      }
      if(!nodes[i]->IsGhost() && nodes[i]->pt_cnt[1]){
        n_list_trg.push_back(nodes[i]);
      }
    }
  }
  size_t node_cnt=std::max(n_list_src.size(),n_list_trg.size());

  std::vector<Mat_Type> type_lst;
  std::vector<std::vector<Node_t*>*> type_node_lst;
  type_lst.push_back(S2U_Type); type_node_lst.push_back(&n_list_src);
  type_lst.push_back(U2U_Type); type_node_lst.push_back(&n_list_src);
  type_lst.push_back(D2D_Type); type_node_lst.push_back(&n_list_trg);
  type_lst.push_back(D2T_Type); type_node_lst.push_back(&n_list_trg);
  type_lst.push_back(U0_Type ); type_node_lst.push_back(&n_list_trg);
  type_lst.push_back(U1_Type ); type_node_lst.push_back(&n_list_trg);
  type_lst.push_back(U2_Type ); type_node_lst.push_back(&n_list_trg);
  type_lst.push_back(W_Type  ); type_node_lst.push_back(&n_list_trg);
  type_lst.push_back(X_Type  ); type_node_lst.push_back(&n_list_trg);
  type_lst.push_back(V1_Type ); type_node_lst.push_back(&n_list_trg);
  std::vector<size_t> interac_cnt(type_lst.size());
  std::vector<size_t> interac_dsp(type_lst.size(),0);
  for(size_t i=0;i<type_lst.size();i++){
    interac_cnt[i]=interac_list.ListCount(type_lst[i]);
  }
  omp_par::scan(&interac_cnt[0],&interac_dsp[0],type_lst.size());
  node_interac_lst.ReInit(node_cnt,interac_cnt.back()+interac_dsp.back());

  // Build interaction lists.
  int omp_p=omp_get_max_threads();
  #pragma omp parallel for
  for(int j=0;j<omp_p;j++){
    for(size_t k=0;k<type_lst.size();k++){
      std::vector<Node_t*>& n_list=*type_node_lst[k];
      size_t a=(n_list.size()*(j  ))/omp_p;
      size_t b=(n_list.size()*(j+1))/omp_p;
      for(size_t i=a;i<b;i++){
        Node_t* n=n_list[i];
        n->interac_list[type_lst[k]].ReInit(interac_cnt[k],&node_interac_lst[i][interac_dsp[k]],false);
        interac_list.BuildList(n,type_lst[k]);
      }
    }
  }
}

template <class FMM_Mat_t>
void FMM_Tree<FMM_Mat_t>::DownwardPass() {
  Profile::Tic("Setup",true,3);
  std::vector<Node_t*> leaf_nodes;
  int max_depth=0;
  { // Build leaf node list
    int max_depth_loc=0;
    std::vector<Node_t*>& nodes=this->GetNodeList();
    for(size_t i=0;i<nodes.size();i++){
      Node_t* n=nodes[i];
      if(!n->IsGhost() && n->IsLeaf()) leaf_nodes.push_back(n);
      if(n->Depth()>max_depth_loc) max_depth_loc=n->Depth();
    }
    max_depth = max_depth_loc;
  }
  Profile::Toc();

  if(bndry==Periodic){ //Add contribution from periodic infinite tiling.
    Profile::Tic("BoundaryCondition",false,5);
    fmm_mat->PeriodicBC(dynamic_cast<Node_t*>(this->RootNode()));
    Profile::Toc();
  }

  for(size_t i=0; i<=(fmm_mat->ScaleInvar()?0:max_depth); i++){ // U,V,W,X-lists

    if(!fmm_mat->ScaleInvar()){ // Precomp
      std::stringstream level_str;
      level_str<<"Level-"<<std::setfill('0')<<std::setw(2)<<i<<"\0";
      Profile::Tic(level_str.str().c_str(),false,5);

      Profile::Tic("Precomp",false,5);
      {// Precomp U
        Profile::Tic("Precomp-U",false,10);
        fmm_mat->SetupPrecomp(setup_data[i+MAX_DEPTH*0]);
        Profile::Toc();
      }
      {// Precomp W
        Profile::Tic("Precomp-W",false,10);
        fmm_mat->SetupPrecomp(setup_data[i+MAX_DEPTH*1]);
        Profile::Toc();
      }
      {// Precomp X
        Profile::Tic("Precomp-X",false,10);
        fmm_mat->SetupPrecomp(setup_data[i+MAX_DEPTH*2]);
        Profile::Toc();
      }
      if(0){// Precomp V
        Profile::Tic("Precomp-V",false,10);
        fmm_mat->SetupPrecomp(setup_data[i+MAX_DEPTH*3]);
        Profile::Toc();
      }
      Profile::Toc();
    }

    {// X-List
      Profile::Tic("X-List",false,5);
      fmm_mat->X_List(setup_data[i+MAX_DEPTH*2]);
      Profile::Toc();
    }

    {// W-List
      Profile::Tic("W-List",false,5);
      fmm_mat->W_List(setup_data[i+MAX_DEPTH*1]);
      Profile::Toc();
    }

    {// U-List
      Profile::Tic("U-List",false,5);
      fmm_mat->U_List(setup_data[i+MAX_DEPTH*0]);
      Profile::Toc();
    }

    {// V-List
      Profile::Tic("V-List",false,5);
      fmm_mat->V_List(setup_data[i+MAX_DEPTH*3]);
      Profile::Toc();
    }

    if(!fmm_mat->ScaleInvar()){ // Wait
      Profile::Toc();
    }
  }

  Profile::Tic("D2D",false,5);
  for(size_t i=0; i<=max_depth; i++){ // Down2Down
    if(!fmm_mat->ScaleInvar()) fmm_mat->SetupPrecomp(setup_data[i+MAX_DEPTH*4]);
    fmm_mat->Down2Down(setup_data[i+MAX_DEPTH*4]);
  }
  Profile::Toc();

  Profile::Tic("D2T",false,5);
  for(int i=0; i<=(fmm_mat->ScaleInvar()?0:max_depth); i++){ // Down2Target
    if(!fmm_mat->ScaleInvar()) fmm_mat->SetupPrecomp(setup_data[i+MAX_DEPTH*5]);
    fmm_mat->Down2Target(setup_data[i+MAX_DEPTH*5]);
  }
  Profile::Toc();

  Profile::Tic("PostProc",false,5);
  typedef typename FMM_Mat_t::FMMTree_t MatTree_t;
  fmm_mat->PostProcessing((MatTree_t*)this, leaf_nodes, bndry);
  Profile::Toc();
}


template <class FMM_Mat_t>
void FMM_Tree<FMM_Mat_t>::Copy_FMMOutput() {
  std::vector<Node_t*>& all_nodes=this->GetNodeList();
  int omp_p=omp_get_max_threads();

  // Copy output to the tree.
  {
    size_t k=all_nodes.size();
    #pragma omp parallel for
    for(int j=0;j<omp_p;j++){
      size_t a=(k*j)/omp_p;
      size_t b=(k*(j+1))/omp_p;
      fmm_mat->CopyOutput(&(all_nodes[a]),b-a);
    }
  }
}

}//end namespace
