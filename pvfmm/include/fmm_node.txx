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
bool FMM_Node<Node>::SubdivCond(){
  int n=(1UL<<this->Dim());
  // Do not subdivide beyond max_depth
  if(this->Depth()>=this->max_depth-1) return false;
  if(!this->IsLeaf()){ // If has non-leaf children, then return true.
    for(int i=0;i<n;i++){
      MPI_Node<Real_t>* ch=static_cast<MPI_Node<Real_t>*>(this->Child(i));
      assert(ch!=NULL); //This should never happen
      if(!ch->IsLeaf() || ch->IsGhost()) return true;
    }
  }
  else{ // Do not refine ghost leaf nodes.
    if(this->IsGhost()) return false;
  }
  if(Node::SubdivCond()) return true;

  if(!this->IsLeaf()){
    size_t pt_vec_size=0;
    for(int i=0;i<n;i++){
      FMM_Node<Node>* ch=static_cast<FMM_Node<Node>*>(this->Child(i));
      pt_vec_size+=ch->src_coord.Dim();
      pt_vec_size+=ch->surf_coord.Dim();
      pt_vec_size+=ch->trg_coord.Dim();
    }
    return pt_vec_size/this->Dim()>this->max_pts;
  }else{
    size_t pt_vec_size=0;
    pt_vec_size+=src_coord.Dim();
    pt_vec_size+=surf_coord.Dim();
    pt_vec_size+=trg_coord.Dim();
    return pt_vec_size/this->Dim()>this->max_pts;
  }
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
