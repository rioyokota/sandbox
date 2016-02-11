namespace pvfmm{

MPI_Node::~MPI_Node(){}

void MPI_Node::Initialize(TreeNode* parent_,int path2node_, TreeNode::NodeData* data_){
  TreeNode::Initialize(parent_,path2node_,data_);

  //Set node coordinates.
  Real_t coord_offset=((Real_t)1.0)/((Real_t)(((uint64_t)1)<<Depth()));
  if(!parent_){
    for(int j=0;j<dim;j++) coord[j]=0;
  }else if(parent_){
    int flag=1;
    for(int j=0;j<dim;j++){
      coord[j]=(parent_)->coord[j]+
               ((Path2Node() & flag)?coord_offset:0.0f);
      flag=flag<<1;
    }
  }

  //Initialize colleagues array.
  int n=pvfmm::pow<unsigned int>(3,Dim());
  for(int i=0;i<n;i++) colleague[i]=NULL;

  //Set MPI_Node specific data.
  typename TreeNode::NodeData* mpi_data=dynamic_cast<typename TreeNode::NodeData*>(data_);
  if(data_){
    max_pts =mpi_data->max_pts;
    pt_coord=mpi_data->coord;
    pt_value=mpi_data->value;
  }else if(parent){
    max_pts =parent->max_pts;
    SetGhost(parent->IsGhost());
  }
}

void MPI_Node::Subdivide(){
  if(!this->IsLeaf()) return;
  TreeNode::Subdivide();
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
    Real_t s=pvfmm::pow<Real_t>(0.5,this->Depth()+1);
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
};

void MPI_Node::ReadVal(std::vector<Real_t> x,std::vector<Real_t> y, std::vector<Real_t> z, Real_t* val, bool show_ghost){
  if(!pt_coord.Dim()) return;
  size_t n_pts=pt_coord.Dim()/dim;
  size_t data_dof=pt_value.Dim()/n_pts;
  std::vector<Real_t> v(data_dof,0);
  for(size_t i=0;i<n_pts;i++)
    for(int j=0;j<data_dof;j++)
      v[j]+=pt_value[i*data_dof+j];
  for(int j=0;j<data_dof;j++)
    v[j]=v[j]/n_pts;
  for(size_t i=0;i<x.size()*y.size()*z.size()*data_dof;i++){
    val[i]=v[i%data_dof];
  }
}


}//end namespace
