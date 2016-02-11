#include <mpi_node.hpp>
//#include <fmm_pts.hpp>

#ifndef _PVFMM_FMM_NODE_HPP_
#define _PVFMM_FMM_NODE_HPP_

namespace pvfmm{

class FMM_Node: public MPI_Node {

 private:

  FMM_Data<Real_t>* fmm_data;

 public:

  Vector<Real_t> src_coord;
  Vector<Real_t> src_value;
  Vector<size_t> src_scatter;
  Vector<Real_t> surf_coord;
  Vector<Real_t> surf_value;
  Vector<size_t> surf_scatter;
  Vector<Real_t> trg_coord;
  Vector<Real_t> trg_value;
  Vector<size_t> trg_scatter;
  size_t pt_cnt[2]; // Number of source, target pts.
  Vector<FMM_Node*> interac_list[Type_Count];

  class NodeData: public MPI_Node::NodeData{
    public:
     Vector<Real_t> src_coord;
     Vector<Real_t> src_value;
     Vector<Real_t> surf_coord;
     Vector<Real_t> surf_value;
     Vector<Real_t> trg_coord;
     Vector<Real_t> trg_value;
  };

  FMM_Node(){
    MPI_Node();
    fmm_data=NULL;
  }

  virtual ~FMM_Node(){
    if(fmm_data!=NULL) mem::aligned_delete(fmm_data);
    fmm_data=NULL;
  }

  virtual void Initialize(MPI_Node* parent_, int path2node_, MPI_Node::NodeData* data_){
    parent=parent_;
    depth=(parent==NULL?0:parent->depth+1);
    if(data_!=NULL){
      dim=data_->dim;
      max_depth=data_->max_depth;
      if(max_depth>MAX_DEPTH) max_depth=MAX_DEPTH;
    }else if(parent!=NULL){
      dim=parent->Dim();
      max_depth=parent->max_depth;
    }
    assert(path2node_>=0 && path2node_<(int)(1U<<dim));
    path2node=path2node_;

    Real_t coord_offset=((Real_t)1.0)/((Real_t)(((uint64_t)1)<<depth));
    if(!parent_){
      for(int j=0;j<dim;j++) coord[j]=0;
    }else if(parent_){
      int flag=1;
      for(int j=0;j<dim;j++){
	coord[j]=parent_->coord[j]+
	  ((Path2Node() & flag)?coord_offset:0.0f);
	flag=flag<<1;
      }
    }

    int n=pvfmm::pow<unsigned int>(3,Dim());
    for(int i=0;i<n;i++) colleague[i]=NULL;

    NodeData* mpi_data=dynamic_cast<NodeData*>(data_);
    if(data_){
      max_pts =mpi_data->max_pts;
      pt_coord=mpi_data->coord;
      pt_value=mpi_data->value;
    }else if(parent){
      max_pts =parent->max_pts;
      SetGhost(parent->IsGhost());
    }

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

  virtual void NodeDataVec(std::vector<Vector<Real_t>*>& coord,
                           std::vector<Vector<Real_t>*>& value,
                           std::vector<Vector<size_t>*>& scatter){
    MPI_Node::NodeDataVec(coord, value, scatter);
    coord  .push_back(&src_coord  );
    value  .push_back(&src_value  );
    scatter.push_back(&src_scatter);
    coord  .push_back(&surf_coord  );
    value  .push_back(&surf_value  );
    scatter.push_back(&surf_scatter);
    coord  .push_back(&trg_coord  );
    value  .push_back(&trg_value  );
    scatter.push_back(&trg_scatter);
  }

  virtual void ClearData(){
    ClearFMMData();
    MPI_Node::ClearData();
  }

  virtual void ClearFMMData(){
    if(fmm_data!=NULL)
      fmm_data->Clear();
  }

  virtual void Truncate(){
    MPI_Node::Truncate();
  }

  FMM_Data<Real_t>*& FMMData(){return fmm_data;}

  MPI_Node* NewNode(MPI_Node* n_=NULL) {
    FMM_Node* n=(n_==NULL?mem::aligned_new<FMM_Node>():static_cast<FMM_Node*>(n_));
    if(fmm_data!=NULL) n->fmm_data=fmm_data->NewData();
    n->dim=dim;
    n->max_depth=max_depth;
    n->max_pts=max_pts;
    return n;
  }

  virtual void Subdivide(){
    if(!this->IsLeaf()) return;
    if(child) return;
    SetStatus(1);
    int n=(1UL<<dim);
    child=mem::aligned_new<MPI_Node*>(n);
    for(int i=0;i<n;i++){
      child[i]=NewNode();
      child[i]->parent=this;
      ((FMM_Node*)child[i])->Initialize(this,i,NULL);
    }
    int nchld=(1UL<<this->Dim());
    if(!IsGhost()){ // Partition point coordinates and values.
      std::vector<Vector<Real_t>*> pt_c;
      std::vector<Vector<Real_t>*> pt_v;
      std::vector<Vector<size_t>*> pt_s;
      this->NodeDataVec(pt_c, pt_v, pt_s);

      std::vector<std::vector<Vector<Real_t>*> > chld_pt_c(nchld);
      std::vector<std::vector<Vector<Real_t>*> > chld_pt_v(nchld);
      std::vector<std::vector<Vector<size_t>*> > chld_pt_s(nchld);
      for(size_t i=0;i<nchld;i++){
	((MPI_Node*)this->Child(i))->NodeDataVec(chld_pt_c[i], chld_pt_v[i], chld_pt_s[i]);
      }

      Real_t* c=this->Coord();
      Real_t s=pvfmm::pow<Real_t>(0.5,depth+1);
      for(size_t j=0;j<pt_c.size();j++){
	if(!pt_c[j] || !pt_c[j]->Dim()) continue;
	Vector<Real_t>& coord=*pt_c[j];
	size_t npts=coord.Dim()/this->dim;

	Vector<size_t> cdata(nchld+1);
	for(size_t i=0;i<nchld+1;i++){
	  long long pt1=-1, pt2=npts;
	  while(pt2-pt1>1){ // binary search
	    long long pt3=(pt1+pt2)/2;
	    assert(pt3<npts);
	    if(pt3<0) pt3=0;
	    int ch_id=(coord[pt3*3+0]>=c[0]+s)*1+
	      (coord[pt3*3+1]>=c[1]+s)*2+
	      (coord[pt3*3+2]>=c[2]+s)*4;
	    if(ch_id< i) pt1=pt3;
	    if(ch_id>=i) pt2=pt3;
	  }
	  cdata[i]=pt2;
	}

	if(pt_c[j]){
	  Vector<Real_t>& vec=*pt_c[j];
	  size_t dof=vec.Dim()/npts;
	  if(dof>0) for(size_t i=0;i<nchld;i++){
	      Vector<Real_t>& chld_vec=*chld_pt_c[i][j];
	      chld_vec.ReInit((cdata[i+1]-cdata[i])*dof, &vec[0]+cdata[i]*dof);
	    }
	  vec.ReInit(0);
	}
	if(pt_v[j]){
	  Vector<Real_t>& vec=*pt_v[j];
	  size_t dof=vec.Dim()/npts;
	  if(dof>0) for(size_t i=0;i<nchld;i++){
	      Vector<Real_t>& chld_vec=*chld_pt_v[i][j];
	      chld_vec.ReInit((cdata[i+1]-cdata[i])*dof, &vec[0]+cdata[i]*dof);
	    }
	  vec.ReInit(0);
	}
	if(pt_s[j]){
	  Vector<size_t>& vec=*pt_s[j];
	  size_t dof=vec.Dim()/npts;
	  if(dof>0) for(size_t i=0;i<nchld;i++){
	      Vector<size_t>& chld_vec=*chld_pt_s[i][j];
	      chld_vec.ReInit((cdata[i+1]-cdata[i])*dof, &vec[0]+cdata[i]*dof);
	    }
	  vec.ReInit(0);
	}
      }
    }
  }

};

}//end namespace

#endif //_PVFMM_FMM_NODE_HPP_
