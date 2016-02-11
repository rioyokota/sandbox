#include <cassert>
#include <mem_mgr.hpp>
#include <mpi_node.hpp>

namespace pvfmm{

FMM_Node::~FMM_Node(){
  if(fmm_data!=NULL) mem::aligned_delete(fmm_data);
  fmm_data=NULL;
}

void FMM_Node::Initialize(TreeNode* parent_,int path2node_, MPI_Node::NodeData* data_){
  MPI_Node::Initialize(parent_,path2node_,data_);

  //Set FMM_Node specific data.
  typename FMM_Node::NodeData* data=dynamic_cast<typename FMM_Node::NodeData*>(data_);
  if(data_!=NULL){
    src_coord=data->src_coord;
    src_value=data->src_value;

    surf_coord=data->surf_coord;
    surf_value=data->surf_value;

    trg_coord=data->trg_coord;
    trg_value=data->trg_value;
  }
}


void FMM_Node::ClearData(){
  ClearFMMData();
  MPI_Node::ClearData();
}


void FMM_Node::ClearFMMData(){
  if(fmm_data!=NULL)
    fmm_data->Clear();
}


TreeNode* FMM_Node::NewNode(TreeNode* n_){
  FMM_Node* n=(n_==NULL?mem::aligned_new<FMM_Node>():static_cast<FMM_Node*>(n_));
  if(fmm_data!=NULL) n->fmm_data=fmm_data->NewData();
  n->dim=dim;
  n->max_depth=max_depth;
  n->max_pts=max_pts;
  return n;
}

void FMM_Node::Subdivide(){
  if(!this->IsLeaf()) return;
  MPI_Node::Subdivide();
}


void FMM_Node::Truncate(){
  MPI_Node::Truncate();
}

}//end namespace
