#include <cassert>
#include <mem_mgr.hpp>
#include <mpi_node.hpp>

namespace pvfmm{

template <class Node>
FMM_Node<Node>::~FMM_Node(){
  if(fmm_data!=NULL) mem::aligned_delete(fmm_data);
  fmm_data=NULL;
}

template <class Node>
void FMM_Node<Node>::Initialize(TreeNode* parent_,int path2node_, TreeNode::NodeData* data_){
  Node::Initialize(parent_,path2node_,data_);

  //Set FMM_Node specific data.
  typename FMM_Node<Node>::NodeData* data=dynamic_cast<typename FMM_Node<Node>::NodeData*>(data_);
  if(data_!=NULL){
    src_coord=data->src_coord;
    src_value=data->src_value;

    surf_coord=data->surf_coord;
    surf_value=data->surf_value;

    trg_coord=data->trg_coord;
    trg_value=data->trg_value;
  }
}


template <class Node>
void FMM_Node<Node>::ClearData(){
  ClearFMMData();
  TreeNode::ClearData();
}


template <class Node>
void FMM_Node<Node>::ClearFMMData(){
  if(fmm_data!=NULL)
    fmm_data->Clear();
}


template <class Node>
TreeNode* FMM_Node<Node>::NewNode(TreeNode* n_){
  FMM_Node<Node>* n=(n_==NULL?mem::aligned_new<FMM_Node<Node> >():static_cast<FMM_Node<Node>*>(n_));
  if(fmm_data!=NULL) n->fmm_data=fmm_data->NewData();
  return Node_t::NewNode(n);
}

template <class Node>
void FMM_Node<Node>::Subdivide(){
  if(!this->IsLeaf()) return;
  Node::Subdivide();
}


template <class Node>
void FMM_Node<Node>::Truncate(){
  Node::Truncate();
}

}//end namespace
