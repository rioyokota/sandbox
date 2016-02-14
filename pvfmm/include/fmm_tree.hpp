#ifndef _PVFMM_FMM_TREE_HPP_
#define _PVFMM_FMM_TREE_HPP_

#include <fmm_pts.hpp>

namespace pvfmm{

template <class FMMNode_t>
class FMM_Tree : public FMM_Pts<FMMNode_t> {

 private:

  inline int p2oLocal(Vector<MortonId> & nodes, Vector<MortonId>& leaves,
		      unsigned int maxNumPts, unsigned int maxDepth, bool complete) {
    assert(maxDepth<=MAX_DEPTH);
    std::vector<MortonId> leaves_lst;
    unsigned int init_size=leaves.Dim();
    unsigned int num_pts=nodes.Dim();
    MortonId curr_node=leaves[0];
    MortonId last_node=leaves[init_size-1].NextId();
    MortonId next_node;
    unsigned int curr_pt=0;
    unsigned int next_pt=curr_pt+maxNumPts;
    while(next_pt <= num_pts){
      next_node = curr_node.NextId();
      while( next_pt < num_pts && next_node > nodes[next_pt] && curr_node.GetDepth() < maxDepth-1 ){
	curr_node = curr_node.getDFD(curr_node.GetDepth()+1);
	next_node = curr_node.NextId();
      }
      leaves_lst.push_back(curr_node);
      curr_node = next_node;
      unsigned int inc=maxNumPts;
      while(next_pt < num_pts && curr_node > nodes[next_pt]){
	inc=inc<<1;
	next_pt+=inc;
	if(next_pt > num_pts){
	  next_pt = num_pts;
	  break;
	}
      }
      curr_pt = std::lower_bound(&nodes[0]+curr_pt,&nodes[0]+next_pt,curr_node,std::less<MortonId>())-&nodes[0];
      if(curr_pt >= num_pts) break;
      next_pt = curr_pt + maxNumPts;
      if(next_pt > num_pts) next_pt = num_pts;
    }
#ifndef NDEBUG
    for(size_t i=0;i<leaves_lst.size();i++){
      size_t a=std::lower_bound(&nodes[0],&nodes[0]+nodes.Dim(),leaves_lst[i],std::less<MortonId>())-&nodes[0];
      size_t b=std::lower_bound(&nodes[0],&nodes[0]+nodes.Dim(),leaves_lst[i].NextId(),std::less<MortonId>())-&nodes[0];
      assert(b-a<=maxNumPts || leaves_lst[i].GetDepth()==maxDepth-1);
      if(i==leaves_lst.size()-1) assert(b==nodes.Dim() && a<nodes.Dim());
      if(i==0) assert(a==0);
    }
#endif
    if(complete) {
      while(curr_node<last_node){
	while( curr_node.NextId() > last_node && curr_node.GetDepth() < maxDepth-1 )
	  curr_node = curr_node.getDFD(curr_node.GetDepth()+1);
	leaves_lst.push_back(curr_node);
	curr_node = curr_node.NextId();
      }
    }
    leaves=leaves_lst;
    return 0;
  }

  inline int points2Octree(const Vector<MortonId>& pt_mid, Vector<MortonId>& nodes,
			   unsigned int maxDepth, unsigned int maxNumPts) {
    int myrank=0, np=1;
    Profile::Tic("SortMortonId", true, 10);
    Vector<MortonId> pt_sorted;
    par::HyperQuickSort(pt_mid, pt_sorted);
    size_t pt_cnt=pt_sorted.Dim();
    Profile::Toc();

    Profile::Tic("p2o_local", false, 10);
    Vector<MortonId> nodes_local(1); nodes_local[0]=MortonId();
    p2oLocal(pt_sorted, nodes_local, maxNumPts, maxDepth, myrank==np-1);
    Profile::Toc();

    Profile::Tic("RemoveDuplicates", true, 10);
    {
      size_t node_cnt=nodes_local.Dim();
      MortonId first_node;
      MortonId  last_node=nodes_local[node_cnt-1];
      size_t i=0;
      std::vector<MortonId> node_lst;
      if(myrank){
	while(i<node_cnt && nodes_local[i].getDFD(maxDepth)<first_node) i++; assert(i);
	last_node=nodes_local[i>0?i-1:0].NextId();

	while(first_node<last_node){
	  while(first_node.isAncestor(last_node))
	    first_node=first_node.getDFD(first_node.GetDepth()+1);
	  if(first_node==last_node) break;
	  node_lst.push_back(first_node);
	  first_node=first_node.NextId();
	}
      }
      for(;i<node_cnt-(myrank==np-1?0:1);i++) node_lst.push_back(nodes_local[i]);
      nodes=node_lst;
    }
    Profile::Toc();

    return 0;
  }

  template <class Real_t>
  void VListHadamard(size_t dof, size_t M_dim, size_t ker_dim0, size_t ker_dim1, Vector<size_t>& interac_dsp,
      Vector<size_t>& interac_vec, Vector<Real_t*>& precomp_mat, Vector<Real_t>& fft_in, Vector<Real_t>& fft_out){
    size_t chld_cnt=1UL<<COORD_DIM;
    size_t fftsize_in =M_dim*ker_dim0*chld_cnt*2;
    size_t fftsize_out=M_dim*ker_dim1*chld_cnt*2;
    Real_t* zero_vec0=mem::aligned_new<Real_t>(fftsize_in );
    Real_t* zero_vec1=mem::aligned_new<Real_t>(fftsize_out);
    size_t n_out=fft_out.Dim()/fftsize_out;
#pragma omp parallel for
    for(size_t k=0;k<n_out;k++){
      Vector<Real_t> dnward_check_fft(fftsize_out, &fft_out[k*fftsize_out], false);
      dnward_check_fft.SetZero();
    }
    size_t mat_cnt=precomp_mat.Dim();
    size_t blk1_cnt=interac_dsp.Dim()/mat_cnt;
    const size_t V_BLK_SIZE=V_BLK_CACHE*64/sizeof(Real_t);
    Real_t** IN_ =mem::aligned_new<Real_t*>(2*V_BLK_SIZE*blk1_cnt*mat_cnt);
    Real_t** OUT_=mem::aligned_new<Real_t*>(2*V_BLK_SIZE*blk1_cnt*mat_cnt);
#pragma omp parallel for
    for(size_t interac_blk1=0; interac_blk1<blk1_cnt*mat_cnt; interac_blk1++){
      size_t interac_dsp0 = (interac_blk1==0?0:interac_dsp[interac_blk1-1]);
      size_t interac_dsp1 =                    interac_dsp[interac_blk1  ] ;
      size_t interac_cnt  = interac_dsp1-interac_dsp0;
      for(size_t j=0;j<interac_cnt;j++){
        IN_ [2*V_BLK_SIZE*interac_blk1 +j]=&fft_in [interac_vec[(interac_dsp0+j)*2+0]];
        OUT_[2*V_BLK_SIZE*interac_blk1 +j]=&fft_out[interac_vec[(interac_dsp0+j)*2+1]];
      }
      IN_ [2*V_BLK_SIZE*interac_blk1 +interac_cnt]=zero_vec0;
      OUT_[2*V_BLK_SIZE*interac_blk1 +interac_cnt]=zero_vec1;
    }
    int omp_p=omp_get_max_threads();
#pragma omp parallel for
    for(int pid=0; pid<omp_p; pid++){
      size_t a=( pid   *M_dim)/omp_p;
      size_t b=((pid+1)*M_dim)/omp_p;
      for(int in_dim=0;in_dim<ker_dim0;in_dim++)
      for(int ot_dim=0;ot_dim<ker_dim1;ot_dim++)
      for(size_t     blk1=0;     blk1<blk1_cnt;    blk1++)
      for(size_t        k=a;        k<       b;       k++)
      for(size_t mat_indx=0; mat_indx< mat_cnt;mat_indx++){
        size_t interac_blk1 = blk1*mat_cnt+mat_indx;
        size_t interac_dsp0 = (interac_blk1==0?0:interac_dsp[interac_blk1-1]);
        size_t interac_dsp1 =                    interac_dsp[interac_blk1  ] ;
        size_t interac_cnt  = interac_dsp1-interac_dsp0;
        Real_t** IN = IN_ + 2*V_BLK_SIZE*interac_blk1;
        Real_t** OUT= OUT_+ 2*V_BLK_SIZE*interac_blk1;
        Real_t* M = precomp_mat[mat_indx] + k*chld_cnt*chld_cnt*2 + (ot_dim+in_dim*ker_dim1)*M_dim*128;
        {
          for(size_t j=0;j<interac_cnt;j+=2){
            Real_t* M_   = M;
            Real_t* IN0  = IN [j+0] + (in_dim*M_dim+k)*chld_cnt*2;
            Real_t* IN1  = IN [j+1] + (in_dim*M_dim+k)*chld_cnt*2;
            Real_t* OUT0 = OUT[j+0] + (ot_dim*M_dim+k)*chld_cnt*2;
            Real_t* OUT1 = OUT[j+1] + (ot_dim*M_dim+k)*chld_cnt*2;
#ifdef __SSE__
            if (j+2 < interac_cnt) {
              _mm_prefetch(((char *)(IN[j+2] + (in_dim*M_dim+k)*chld_cnt*2)), _MM_HINT_T0);
              _mm_prefetch(((char *)(IN[j+2] + (in_dim*M_dim+k)*chld_cnt*2) + 64), _MM_HINT_T0);
              _mm_prefetch(((char *)(IN[j+3] + (in_dim*M_dim+k)*chld_cnt*2)), _MM_HINT_T0);
              _mm_prefetch(((char *)(IN[j+3] + (in_dim*M_dim+k)*chld_cnt*2) + 64), _MM_HINT_T0);
              _mm_prefetch(((char *)(OUT[j+2] + (ot_dim*M_dim+k)*chld_cnt*2)), _MM_HINT_T0);
              _mm_prefetch(((char *)(OUT[j+2] + (ot_dim*M_dim+k)*chld_cnt*2) + 64), _MM_HINT_T0);
              _mm_prefetch(((char *)(OUT[j+3] + (ot_dim*M_dim+k)*chld_cnt*2)), _MM_HINT_T0);
              _mm_prefetch(((char *)(OUT[j+3] + (ot_dim*M_dim+k)*chld_cnt*2) + 64), _MM_HINT_T0);
            }
#endif
            matmult_8x8x2(M_, IN0, IN1, OUT0, OUT1);
          }
        }
      }
    }
    {
      Profile::Add_FLOP(8*8*8*(interac_vec.Dim()/2)*M_dim*ker_dim0*ker_dim1*dof);
    }
    mem::aligned_delete<Real_t*>(IN_ );
    mem::aligned_delete<Real_t*>(OUT_);
    mem::aligned_delete<Real_t>(zero_vec0);
    mem::aligned_delete<Real_t>(zero_vec1);
  }

  template<typename ElemType>
  void CopyVec(std::vector<std::vector<ElemType> >& vec_, pvfmm::Vector<ElemType>& vec) {
    int omp_p=omp_get_max_threads();
    std::vector<size_t> vec_dsp(omp_p+1,0);
    for(size_t tid=0;tid<omp_p;tid++){
      vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
    }
    vec.ReInit(vec_dsp[omp_p]);
#pragma omp parallel for
    for(size_t tid=0;tid<omp_p;tid++){
      memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
    }
  }

 public:

  typedef typename FMM_Pts<FMMNode_t>::FMMData FMMData_t;
  typedef typename FMM_Pts<FMMNode_t>::ptSetupData ptSetupData;
  typedef typename FMM_Pts<FMMNode_t>::PackedData PackedData;
  typedef typename FMM_Pts<FMMNode_t>::InteracData InteracData;
  typedef FMM_Tree FMMTree_t;
  using FMM_Pts<FMMNode_t>::kernel;
  using FMM_Pts<FMMNode_t>::PtSetup;
  using FMM_Pts<FMMNode_t>::Precomp;
  using FMM_Pts<FMMNode_t>::MultipoleOrder;
  using FMM_Pts<FMMNode_t>::FFT_UpEquiv;
  using FMM_Pts<FMMNode_t>::SetupInterac;
  using FMM_Pts<FMMNode_t>::FFT_Check2Equiv;
  using FMM_Pts<FMMNode_t>::EvalList;

  int dim;
  FMMNode_t* root_node;
  int max_depth;
  std::vector<FMMNode_t*> node_lst;
  mem::MemoryManager memgr;

  std::vector<Matrix<Real_t> > node_data_buff;
  pvfmm::Matrix<FMMNode_t*> node_interac_lst;
  InteracList<FMMNode_t> interac_list;
  BoundaryType bndry;
  std::vector<Matrix<char> > precomp_lst;
  std::vector<SetupData<Real_t,FMMNode_t> > setup_data;
  std::vector<Vector<Real_t> > upwd_check_surf;
  std::vector<Vector<Real_t> > upwd_equiv_surf;
  std::vector<Vector<Real_t> > dnwd_check_surf;
  std::vector<Vector<Real_t> > dnwd_equiv_surf;

  FMM_Tree(): dim(0), root_node(NULL), max_depth(MAX_DEPTH), memgr(0), bndry(FreeSpace) { };

  ~FMM_Tree(){
    if(RootNode()!=NULL){
      mem::aligned_delete(root_node);
    }
  }

  void Initialize(typename FMM_Node::NodeData* init_data) {
    Profile::Tic("InitTree",true);{
      Profile::Tic("InitRoot",false,5);
      dim=init_data->dim;
      max_depth=init_data->max_depth;
      if(max_depth>MAX_DEPTH) max_depth=MAX_DEPTH;
      if(root_node) mem::aligned_delete(root_node);
      root_node=mem::aligned_new<FMMNode_t>();
      root_node->Initialize(NULL,0,init_data);
      FMMNode_t* rnode=this->RootNode();
      assert(this->dim==COORD_DIM);
      Profile::Toc();
  
      Profile::Tic("Points2Octee",true,5);
      Vector<MortonId> lin_oct;
      {
        Vector<MortonId> pt_mid;
        Vector<Real_t>& pt_c=rnode->pt_coord;
        size_t pt_cnt=pt_c.Dim()/this->dim;
        pt_mid.Resize(pt_cnt);
#pragma omp parallel for
        for(size_t i=0;i<pt_cnt;i++){
        pt_mid[i]=MortonId(pt_c[i*COORD_DIM+0],pt_c[i*COORD_DIM+1],pt_c[i*COORD_DIM+2],this->max_depth);
      }
        points2Octree(pt_mid,lin_oct,this->max_depth,init_data->max_pts);
      }
      Profile::Toc();
  
      Profile::Tic("ScatterPoints",true,5);
      {
        std::vector<Vector<Real_t>*> coord_lst;
        std::vector<Vector<Real_t>*> value_lst;
        std::vector<Vector<size_t>*> scatter_lst;
        rnode->NodeDataVec(coord_lst, value_lst, scatter_lst);
        assert(coord_lst.size()==value_lst.size());
        assert(coord_lst.size()==scatter_lst.size());
  
        Vector<MortonId> pt_mid;
        Vector<size_t> scatter_index;
        for(size_t i=0;i<coord_lst.size();i++){
          if(!coord_lst[i]) continue;
          Vector<Real_t>& pt_c=*coord_lst[i];
          size_t pt_cnt=pt_c.Dim()/this->dim;
          pt_mid.Resize(pt_cnt);
#pragma omp parallel for
          for(size_t i=0;i<pt_cnt;i++){
    	  pt_mid[i]=MortonId(pt_c[i*COORD_DIM+0],pt_c[i*COORD_DIM+1],pt_c[i*COORD_DIM+2],this->max_depth);
          }
          par::SortScatterIndex(pt_mid  , scatter_index, &lin_oct[0]);
          par::ScatterForward  (pt_c, scatter_index);
          if(value_lst[i]!=NULL){
            Vector<Real_t>& pt_v=*value_lst[i];
            par::ScatterForward(pt_v, scatter_index);
          }
          if(scatter_lst[i]!=NULL){
            Vector<size_t>& pt_s=*scatter_lst[i];
            pt_s=scatter_index;
          }
        }
      }
      Profile::Toc();
  
      Profile::Tic("PointerTree",false,5);
      {
        int omp_p=omp_get_max_threads();
  
        rnode->SetGhost(false);
        for(int i=0;i<omp_p;i++){
          size_t idx=(lin_oct.Dim()*i)/omp_p;
          FMMNode_t* n=FindNode(lin_oct[idx], true);
          assert(n->GetMortonId()==lin_oct[idx]);
          UNUSED(n);
        }
  
#pragma omp parallel for
        for(int i=0;i<omp_p;i++){
          size_t a=(lin_oct.Dim()* i   )/omp_p;
          size_t b=(lin_oct.Dim()*(i+1))/omp_p;
  
          size_t idx=a;
          FMMNode_t* n=FindNode(lin_oct[idx], false);
          if(a==0) n=rnode;
          while(n!=NULL && (idx<b || i==omp_p-1)){
            n->SetGhost(false);
            MortonId dn=n->GetMortonId();
            if(idx<b && dn.isAncestor(lin_oct[idx])){
              if(n->IsLeaf()) n->Subdivide();
            }else if(idx<b && dn==lin_oct[idx]){
              if(!n->IsLeaf()) n->Truncate();
              assert(n->IsLeaf());
              idx++;
            }else{
              n->Truncate();
              n->SetGhost(true);
            }
            n=this->PreorderNxt(n);
          }
        }
      }Profile::Toc();
      Profile::Tic("InitFMMData",true,5);{
	std::vector<FMMNode_t*>& nodes=this->GetNodeList();
#pragma omp parallel for
	for(size_t i=0;i<nodes.size();i++){
	  if(nodes[i]->FMMData()==NULL) nodes[i]->FMMData()=mem::aligned_new<FMMData_t>();
	}
      }Profile::Toc();   
    }Profile::Toc();
  }

  int Dim() {return dim;}

  FMMNode_t* RootNode() {return root_node;}

  FMMNode_t* PreorderFirst() {return root_node;}

  FMMNode_t* PreorderNxt(FMMNode_t* curr_node) {
    assert(curr_node!=NULL);
    int n=(1UL<<dim);
    if(!curr_node->IsLeaf())
      for(int i=0;i<n;i++)
	if(curr_node->Child(i)!=NULL)
	  return curr_node->Child(i);
    FMMNode_t* node=curr_node;
    while(true){
      int i=node->Path2Node()+1;
      node=node->Parent();
      if(node==NULL) return NULL;
      for(;i<n;i++)
	if(node->Child(i)!=NULL)
	  return node->Child(i);
    }
  }

  void SetColleagues(BoundaryType bndry=FreeSpace, FMMNode_t* node=NULL) {
    int n1=(int)pvfmm::pow<unsigned int>(3,this->Dim());
    int n2=(int)pvfmm::pow<unsigned int>(2,this->Dim());
    if(node==NULL){
      FMMNode_t* curr_node=this->PreorderFirst();
      if(curr_node!=NULL){
        if(bndry==Periodic){
          for(int i=0;i<n1;i++)
            curr_node->SetColleague(curr_node,i);
        }else{
          curr_node->SetColleague(curr_node,(n1-1)/2);
        }
        curr_node=this->PreorderNxt(curr_node);
      }
      Vector<std::vector<FMMNode_t*> > nodes(MAX_DEPTH);
      while(curr_node!=NULL){
        nodes[curr_node->depth].push_back(curr_node);
        curr_node=this->PreorderNxt(curr_node);
      }
      for(size_t i=0;i<MAX_DEPTH;i++){
        size_t j0=nodes[i].size();
        FMMNode_t** nodes_=&nodes[i][0];
#pragma omp parallel for
        for(size_t j=0;j<j0;j++){
          SetColleagues(bndry, nodes_[j]);
        }
      }
    }else{
      FMMNode_t* parent_node;
      FMMNode_t* tmp_node1;
      FMMNode_t* tmp_node2;
      for(int i=0;i<n1;i++)node->SetColleague(NULL,i);
      parent_node=node->Parent();
      if(parent_node==NULL) return;
      int l=node->Path2Node();
      for(int i=0;i<n1;i++){
        tmp_node1=parent_node->Colleague(i);
        if(tmp_node1!=NULL)
        if(!tmp_node1->IsLeaf()){
          for(int j=0;j<n2;j++){
            tmp_node2=tmp_node1->Child(j);
            if(tmp_node2!=NULL){

              bool flag=true;
              int a=1,b=1,new_indx=0;
              for(int k=0;k<this->Dim();k++){
                int indx_diff=(((i/b)%3)-1)*2+((j/a)%2)-((l/a)%2);
                if(-1>indx_diff || indx_diff>1) flag=false;
                new_indx+=(indx_diff+1)*b;
                a*=2;b*=3;
              }
              if(flag){
                node->SetColleague(tmp_node2,new_indx);
              }
            }
          }
        }
      }
    }
  }

  std::vector<FMMNode_t*>& GetNodeList() {
    if(root_node->GetStatus() & 1){
      node_lst.clear();
      FMMNode_t* n=this->PreorderFirst();
      while(n!=NULL){
	int& status=n->GetStatus();
	status=(status & (~(int)1));
	node_lst.push_back(n);
	n=this->PreorderNxt(n);
      }
    }
    return node_lst;
  }

  FMMNode_t* FindNode(MortonId& key, bool subdiv, FMMNode_t* start=NULL) {
    int num_child=1UL<<this->Dim();
    FMMNode_t* n=start;
    if(n==NULL) n=this->RootNode();
    while(n->GetMortonId()<key && (!n->IsLeaf()||subdiv)){
      if(n->IsLeaf() && !n->IsGhost()) n->Subdivide();
      if(n->IsLeaf()) break;
      for(int j=0;j<num_child;j++){
	if(n->Child(j)->GetMortonId().NextId()>key){
	  n=n->Child(j);
	  break;
	}
      }
    }
    assert(!subdiv || n->IsGhost() || n->GetMortonId()==key);
    return n;
  }

  FMMNode_t* PostorderFirst() {
    FMMNode_t* node=root_node;
    int n=(1UL<<dim);
    while(true){
      if(node->IsLeaf()) return node;
      for(int i=0;i<n;i++) {
	if(node->Child(i)!=NULL){
	  node=node->Child(i);
	  break;
	}
      }
    }
  }

  FMMNode_t* PostorderNxt(FMMNode_t* curr_node) {
    assert(curr_node!=NULL);
    FMMNode_t* node=curr_node;
    int j=node->Path2Node()+1;
    node=node->Parent();
    if(node==NULL) return NULL;
    int n=(1UL<<dim);
    for(;j<n;j++){
      if(node->Child(j)!=NULL){
	node=node->Child(j);
	while(true){
	  if(node->IsLeaf()) return node;
	  for(int i=0;i<n;i++) {
	    if(node->Child(i)!=NULL){
	      node=node->Child(i);
	      break;
	    }
	  }
	}
      }
    }
    return node;
  }

  void InitFMM_Tree(bool refine, BoundaryType bndry_=FreeSpace) {
    Profile::Tic("InitFMM_Tree",true);{
      interac_list.Initialize(this->Dim());
      bndry=bndry_;
    }Profile::Toc();
  }

  void SetupFMM() {
    Profile::Tic("SetupFMM",true);{
    Profile::Tic("SetColleagues",false,3);
    this->SetColleagues(bndry);
    Profile::Toc();
    Profile::Tic("CollectNodeData",false,3);
    FMMNode_t* n=dynamic_cast<FMMNode_t*>(this->PostorderFirst());
    std::vector<FMMNode_t*> all_nodes;
    while(n!=NULL){
      n->pt_cnt[0]=0;
      n->pt_cnt[1]=0;
      all_nodes.push_back(n);
      n=static_cast<FMMNode_t*>(this->PostorderNxt(n));
    }
    std::vector<Vector<FMMNode_t*> > node_lists; // TODO: Remove this parameter, not really needed
    this->CollectNodeData(this,all_nodes, node_data_buff, node_lists);
    Profile::Toc();
  
    Profile::Tic("BuildLists",false,3);
    BuildInteracLists();
    Profile::Toc();
    setup_data.resize(8*MAX_DEPTH);
    precomp_lst.resize(8);
    Profile::Tic("UListSetup",false,3);
    for(size_t i=0;i<MAX_DEPTH;i++){
      setup_data[i+MAX_DEPTH*0].precomp_data=&precomp_lst[0];
      this->U_ListSetup(setup_data[i+MAX_DEPTH*0],this,node_data_buff,node_lists,this->ScaleInvar()?(i==0?-1:MAX_DEPTH+1):i);
    }
    Profile::Toc();
    Profile::Tic("WListSetup",false,3);
    for(size_t i=0;i<MAX_DEPTH;i++){
      setup_data[i+MAX_DEPTH*1].precomp_data=&precomp_lst[1];
      this->W_ListSetup(setup_data[i+MAX_DEPTH*1],this,node_data_buff,node_lists,this->ScaleInvar()?(i==0?-1:MAX_DEPTH+1):i);
    }
    Profile::Toc();
    Profile::Tic("XListSetup",false,3);
    for(size_t i=0;i<MAX_DEPTH;i++){
      setup_data[i+MAX_DEPTH*2].precomp_data=&precomp_lst[2];
      this->X_ListSetup(setup_data[i+MAX_DEPTH*2],this,node_data_buff,node_lists,this->ScaleInvar()?(i==0?-1:MAX_DEPTH+1):i);
    }
    Profile::Toc();
    Profile::Tic("VListSetup",false,3);
    for(size_t i=0;i<MAX_DEPTH;i++){
      setup_data[i+MAX_DEPTH*3].precomp_data=&precomp_lst[3];
      this->V_ListSetup(setup_data[i+MAX_DEPTH*3],this,node_data_buff,node_lists,this->ScaleInvar()?(i==0?-1:MAX_DEPTH+1):i);
    }
    Profile::Toc();
    Profile::Tic("D2DSetup",false,3);
    for(size_t i=0;i<MAX_DEPTH;i++){
      setup_data[i+MAX_DEPTH*4].precomp_data=&precomp_lst[4];
      this->Down2DownSetup(setup_data[i+MAX_DEPTH*4],this,node_data_buff,node_lists,i);
    }
    Profile::Toc();
    Profile::Tic("D2TSetup",false,3);
    for(size_t i=0;i<MAX_DEPTH;i++){
      setup_data[i+MAX_DEPTH*5].precomp_data=&precomp_lst[5];
      this->Down2TargetSetup(setup_data[i+MAX_DEPTH*5],this,node_data_buff,node_lists,this->ScaleInvar()?(i==0?-1:MAX_DEPTH+1):i);
    }
    Profile::Toc();
  
    Profile::Tic("S2USetup",false,3);
    for(size_t i=0;i<MAX_DEPTH;i++){
      setup_data[i+MAX_DEPTH*6].precomp_data=&precomp_lst[6];
      this->Source2UpSetup(setup_data[i+MAX_DEPTH*6],this,node_data_buff,node_lists,this->ScaleInvar()?(i==0?-1:MAX_DEPTH+1):i);
    }
    Profile::Toc();
    Profile::Tic("U2USetup",false,3);
    for(size_t i=0;i<MAX_DEPTH;i++){
      setup_data[i+MAX_DEPTH*7].precomp_data=&precomp_lst[7];
      this->Up2UpSetup(setup_data[i+MAX_DEPTH*7],this,node_data_buff,node_lists,i);
    }
    Profile::Toc();
    ClearFMMData();
    }Profile::Toc();
  }
  
  void ClearFMMData() {
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
  
  void RunFMM() {
    Profile::Tic("RunFMM",true);
    {
      Profile::Tic("UpwardPass",false,2);
      UpwardPass();
      Profile::Toc();
      Profile::Tic("DownwardPass",true,2);
      DownwardPass();
      Profile::Toc();
    }
    Profile::Toc();
  }
    
  void UpwardPass() {
    int max_depth=0;
    {
      int max_depth_loc=0;
      std::vector<FMMNode_t*>& nodes=this->GetNodeList();
      for(size_t i=0;i<nodes.size();i++){
        FMMNode_t* n=nodes[i];
        if(n->depth>max_depth_loc) max_depth_loc=n->depth;
      }
      max_depth = max_depth_loc;
    }
    Profile::Tic("S2U",false,5);
    for(int i=0; i<=(this->ScaleInvar()?0:max_depth); i++){
      if(!this->ScaleInvar()) this->SetupPrecomp(setup_data[i+MAX_DEPTH*6]);
      this->Source2Up(setup_data[i+MAX_DEPTH*6]);
    }
    Profile::Toc();
    Profile::Tic("U2U",false,5);
    for(int i=max_depth-1; i>=0; i--){
      if(!this->ScaleInvar()) this->SetupPrecomp(setup_data[i+MAX_DEPTH*7]);
      this->Up2Up(setup_data[i+MAX_DEPTH*7]);
    }
    Profile::Toc();
  }
  
  void BuildInteracLists() {
    std::vector<FMMNode_t*> n_list_src;
    std::vector<FMMNode_t*> n_list_trg;
    {
      std::vector<FMMNode_t*>& nodes=this->GetNodeList();
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
    std::vector<std::vector<FMMNode_t*>*> type_node_lst;
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
    int omp_p=omp_get_max_threads();
#pragma omp parallel for
    for(int j=0;j<omp_p;j++){
      for(size_t k=0;k<type_lst.size();k++){
        std::vector<FMMNode_t*>& n_list=*type_node_lst[k];
        size_t a=(n_list.size()*(j  ))/omp_p;
        size_t b=(n_list.size()*(j+1))/omp_p;
        for(size_t i=a;i<b;i++){
          FMMNode_t* n=n_list[i];
          n->interac_list[type_lst[k]].ReInit(interac_cnt[k],&node_interac_lst[i][interac_dsp[k]],false);
          interac_list.BuildList(n,type_lst[k]);
        }
      }
    }
  }

  void PeriodicBC(FMMNode_t* node){
    if(!this->ScaleInvar() || this->MultipoleOrder()==0) return;
    Matrix<Real_t>& M = Precomp(0, BC_Type, 0);
    assert(node->FMMData()->upward_equiv.Dim()>0);
    int dof=1;
    Vector<Real_t>& upward_equiv=node->FMMData()->upward_equiv;
    Vector<Real_t>& dnward_equiv=node->FMMData()->dnward_equiv;
    assert(upward_equiv.Dim()==M.Dim(0)*dof);
    assert(dnward_equiv.Dim()==M.Dim(1)*dof);
    Matrix<Real_t> d_equiv(dof,M.Dim(1),&dnward_equiv[0],false);
    Matrix<Real_t> u_equiv(dof,M.Dim(0),&upward_equiv[0],false);
    Matrix<Real_t>::GEMM(d_equiv,u_equiv,M);
  }

  void V_ListSetup(SetupData<Real_t,FMMNode_t>&  setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level){
    if(!this->MultipoleOrder()) return;
    if(level==0) return;
    {
      setup_data.level=level;
      setup_data.kernel=kernel->k_m2l;
      setup_data.interac_type.resize(1);
      setup_data.interac_type[0]=V1_Type;
      setup_data. input_data=&buff[0];
      setup_data.output_data=&buff[1];
      Vector<FMMNode_t*>& nodes_in =n_list[2];
      Vector<FMMNode_t*>& nodes_out=n_list[3];
      setup_data.nodes_in .clear();
      setup_data.nodes_out.clear();
      for(size_t i=0;i<nodes_in .Dim();i++) if((nodes_in [i]->depth==level-1 || level==-1) && nodes_in [i]->pt_cnt[0]) setup_data.nodes_in .push_back(nodes_in [i]);
      for(size_t i=0;i<nodes_out.Dim();i++) if((nodes_out[i]->depth==level-1 || level==-1) && nodes_out[i]->pt_cnt[1]) setup_data.nodes_out.push_back(nodes_out[i]);
    }
    std::vector<FMMNode_t*>& nodes_in =setup_data.nodes_in ;
    std::vector<FMMNode_t*>& nodes_out=setup_data.nodes_out;
    std::vector<Vector<Real_t>*>&  input_vector=setup_data. input_vector;  input_vector.clear();
    std::vector<Vector<Real_t>*>& output_vector=setup_data.output_vector; output_vector.clear();
    for(size_t i=0;i<nodes_in .size();i++)  input_vector.push_back(&(nodes_in[i]->Child(0)->FMMData())->upward_equiv);
    for(size_t i=0;i<nodes_out.size();i++) output_vector.push_back(&(nodes_out[i]->Child(0)->FMMData())->dnward_equiv);
    Real_t eps=1e-10;
    size_t n_in =nodes_in .size();
    size_t n_out=nodes_out.size();
    Profile::Tic("Interac-Data",true,25);
    Matrix<char>& interac_data=setup_data.interac_data;
    if(n_out>0 && n_in >0){
      size_t precomp_offset=0;
      Mat_Type& interac_type=setup_data.interac_type[0];
      size_t mat_cnt=this->interac_list.ListCount(interac_type);
      Matrix<size_t> precomp_data_offset;
      std::vector<size_t> interac_mat;
      std::vector<Real_t*> interac_mat_ptr;
      {
        for(size_t mat_id=0;mat_id<mat_cnt;mat_id++){
          Matrix<Real_t>& M = this->mat->Mat(level, interac_type, mat_id);
          interac_mat_ptr.push_back(&M[0][0]);
        }
      }
      size_t dof;
      size_t m=MultipoleOrder();
      size_t ker_dim0=setup_data.kernel->ker_dim[0];
      size_t ker_dim1=setup_data.kernel->ker_dim[1];
      size_t fftsize;
      {
        size_t n1=m*2;
        size_t n2=n1*n1;
        size_t n3_=n2*(n1/2+1);
        size_t chld_cnt=1UL<<COORD_DIM;
        fftsize=2*n3_*chld_cnt;
        dof=1;
      }
      int omp_p=omp_get_max_threads();
      size_t buff_size=DEVICE_BUFFER_SIZE*1024l*1024l;
      size_t n_blk0=2*fftsize*dof*(ker_dim0*n_in +ker_dim1*n_out)*sizeof(Real_t)/buff_size;
      if(n_blk0==0) n_blk0=1;
      std::vector<std::vector<size_t> >  fft_vec(n_blk0);
      std::vector<std::vector<size_t> > ifft_vec(n_blk0);
      std::vector<std::vector<Real_t> >  fft_scl(n_blk0);
      std::vector<std::vector<Real_t> > ifft_scl(n_blk0);
      std::vector<std::vector<size_t> > interac_vec(n_blk0);
      std::vector<std::vector<size_t> > interac_dsp(n_blk0);
      {
        Matrix<Real_t>&  input_data=*setup_data. input_data;
        Matrix<Real_t>& output_data=*setup_data.output_data;
        std::vector<std::vector<FMMNode_t*> > nodes_blk_in (n_blk0);
        std::vector<std::vector<FMMNode_t*> > nodes_blk_out(n_blk0);
        Vector<Real_t> src_scal=this->kernel->k_m2l->src_scal;
        Vector<Real_t> trg_scal=this->kernel->k_m2l->trg_scal;
  
        for(size_t i=0;i<n_in;i++) nodes_in[i]->node_id=i;
        for(size_t blk0=0;blk0<n_blk0;blk0++){
          size_t blk0_start=(n_out* blk0   )/n_blk0;
          size_t blk0_end  =(n_out*(blk0+1))/n_blk0;
          std::vector<FMMNode_t*>& nodes_in_ =nodes_blk_in [blk0];
          std::vector<FMMNode_t*>& nodes_out_=nodes_blk_out[blk0];
          {
            std::set<FMMNode_t*> nodes_in;
            for(size_t i=blk0_start;i<blk0_end;i++){
              nodes_out_.push_back(nodes_out[i]);
              Vector<FMMNode_t*>& lst=nodes_out[i]->interac_list[interac_type];
              for(size_t k=0;k<mat_cnt;k++) if(lst[k]!=NULL && lst[k]->pt_cnt[0]) nodes_in.insert(lst[k]);
            }
            for(typename std::set<FMMNode_t*>::iterator node=nodes_in.begin(); node != nodes_in.end(); node++){
              nodes_in_.push_back(*node);
            }
            size_t  input_dim=nodes_in_ .size()*ker_dim0*dof*fftsize;
            size_t output_dim=nodes_out_.size()*ker_dim1*dof*fftsize;
            size_t buffer_dim=2*(ker_dim0+ker_dim1)*dof*fftsize*omp_p;
            if(buff_size<(input_dim + output_dim + buffer_dim)*sizeof(Real_t))
              buff_size=(input_dim + output_dim + buffer_dim)*sizeof(Real_t);
          }
          {
            for(size_t i=0;i<nodes_in_ .size();i++) fft_vec[blk0].push_back((size_t)(& input_vector[nodes_in_[i]->node_id][0][0]- input_data[0]));
            for(size_t i=0;i<nodes_out_.size();i++)ifft_vec[blk0].push_back((size_t)(&output_vector[blk0_start   +     i ][0][0]-output_data[0]));
            size_t scal_dim0=src_scal.Dim();
            size_t scal_dim1=trg_scal.Dim();
            fft_scl [blk0].resize(nodes_in_ .size()*scal_dim0);
            ifft_scl[blk0].resize(nodes_out_.size()*scal_dim1);
            for(size_t i=0;i<nodes_in_ .size();i++){
              size_t depth=nodes_in_[i]->depth+1;
              for(size_t j=0;j<scal_dim0;j++){
                fft_scl[blk0][i*scal_dim0+j]=pvfmm::pow<Real_t>(2.0, src_scal[j]*depth);
              }
            }
            for(size_t i=0;i<nodes_out_.size();i++){
              size_t depth=nodes_out_[i]->depth+1;
              for(size_t j=0;j<scal_dim1;j++){
                ifft_scl[blk0][i*scal_dim1+j]=pvfmm::pow<Real_t>(2.0, trg_scal[j]*depth);
              }
            }
          }
        }
        for(size_t blk0=0;blk0<n_blk0;blk0++){
          std::vector<FMMNode_t*>& nodes_in_ =nodes_blk_in [blk0];
          std::vector<FMMNode_t*>& nodes_out_=nodes_blk_out[blk0];
          for(size_t i=0;i<nodes_in_.size();i++) nodes_in_[i]->node_id=i;
          {
            size_t n_blk1=nodes_out_.size()*(2)*sizeof(Real_t)/(64*V_BLK_CACHE);
            if(n_blk1==0) n_blk1=1;
            size_t interac_dsp_=0;
            for(size_t blk1=0;blk1<n_blk1;blk1++){
              size_t blk1_start=(nodes_out_.size()* blk1   )/n_blk1;
              size_t blk1_end  =(nodes_out_.size()*(blk1+1))/n_blk1;
              for(size_t k=0;k<mat_cnt;k++){
                for(size_t i=blk1_start;i<blk1_end;i++){
                  Vector<FMMNode_t*>& lst=nodes_out_[i]->interac_list[interac_type];
                  if(lst[k]!=NULL && lst[k]->pt_cnt[0]){
                    interac_vec[blk0].push_back(lst[k]->node_id*fftsize*ker_dim0*dof);
                    interac_vec[blk0].push_back(    i          *fftsize*ker_dim1*dof);
                    interac_dsp_++;
                  }
                }
                interac_dsp[blk0].push_back(interac_dsp_);
              }
            }
          }
        }
      }
      {
        size_t data_size=sizeof(size_t)*6;
        for(size_t blk0=0;blk0<n_blk0;blk0++){
          data_size+=sizeof(size_t)+    fft_vec[blk0].size()*sizeof(size_t);
          data_size+=sizeof(size_t)+   ifft_vec[blk0].size()*sizeof(size_t);
          data_size+=sizeof(size_t)+    fft_scl[blk0].size()*sizeof(Real_t);
          data_size+=sizeof(size_t)+   ifft_scl[blk0].size()*sizeof(Real_t);
          data_size+=sizeof(size_t)+interac_vec[blk0].size()*sizeof(size_t);
          data_size+=sizeof(size_t)+interac_dsp[blk0].size()*sizeof(size_t);
        }
        data_size+=sizeof(size_t)+interac_mat.size()*sizeof(size_t);
        data_size+=sizeof(size_t)+interac_mat_ptr.size()*sizeof(Real_t*);
        if(data_size>interac_data.Dim(0)*interac_data.Dim(1))
          interac_data.ReInit(1,data_size);
        char* data_ptr=&interac_data[0][0];
        ((size_t*)data_ptr)[0]=buff_size; data_ptr+=sizeof(size_t);
        ((size_t*)data_ptr)[0]=        m; data_ptr+=sizeof(size_t);
        ((size_t*)data_ptr)[0]=      dof; data_ptr+=sizeof(size_t);
        ((size_t*)data_ptr)[0]= ker_dim0; data_ptr+=sizeof(size_t);
        ((size_t*)data_ptr)[0]= ker_dim1; data_ptr+=sizeof(size_t);
        ((size_t*)data_ptr)[0]=   n_blk0; data_ptr+=sizeof(size_t);
        ((size_t*)data_ptr)[0]= interac_mat.size(); data_ptr+=sizeof(size_t);
        mem::memcopy(data_ptr, &interac_mat[0], interac_mat.size()*sizeof(size_t));
        data_ptr+=interac_mat.size()*sizeof(size_t);
        ((size_t*)data_ptr)[0]= interac_mat_ptr.size(); data_ptr+=sizeof(size_t);
        mem::memcopy(data_ptr, &interac_mat_ptr[0], interac_mat_ptr.size()*sizeof(Real_t*));
        data_ptr+=interac_mat_ptr.size()*sizeof(Real_t*);
        for(size_t blk0=0;blk0<n_blk0;blk0++){
          ((size_t*)data_ptr)[0]= fft_vec[blk0].size(); data_ptr+=sizeof(size_t);
          mem::memcopy(data_ptr, & fft_vec[blk0][0],  fft_vec[blk0].size()*sizeof(size_t));
          data_ptr+= fft_vec[blk0].size()*sizeof(size_t);
          ((size_t*)data_ptr)[0]=ifft_vec[blk0].size(); data_ptr+=sizeof(size_t);
          mem::memcopy(data_ptr, &ifft_vec[blk0][0], ifft_vec[blk0].size()*sizeof(size_t));
          data_ptr+=ifft_vec[blk0].size()*sizeof(size_t);
          ((size_t*)data_ptr)[0]= fft_scl[blk0].size(); data_ptr+=sizeof(size_t);
          mem::memcopy(data_ptr, & fft_scl[blk0][0],  fft_scl[blk0].size()*sizeof(Real_t));
          data_ptr+= fft_scl[blk0].size()*sizeof(Real_t);
          ((size_t*)data_ptr)[0]=ifft_scl[blk0].size(); data_ptr+=sizeof(size_t);
          mem::memcopy(data_ptr, &ifft_scl[blk0][0], ifft_scl[blk0].size()*sizeof(Real_t));
          data_ptr+=ifft_scl[blk0].size()*sizeof(Real_t);
          ((size_t*)data_ptr)[0]=interac_vec[blk0].size(); data_ptr+=sizeof(size_t);
          mem::memcopy(data_ptr, &interac_vec[blk0][0], interac_vec[blk0].size()*sizeof(size_t));
          data_ptr+=interac_vec[blk0].size()*sizeof(size_t);
          ((size_t*)data_ptr)[0]=interac_dsp[blk0].size(); data_ptr+=sizeof(size_t);
          mem::memcopy(data_ptr, &interac_dsp[blk0][0], interac_dsp[blk0].size()*sizeof(size_t));
          data_ptr+=interac_dsp[blk0].size()*sizeof(size_t);
        }
      }
    }
    Profile::Toc();
  }
  
  void V_List(SetupData<Real_t,FMMNode_t>&  setup_data){
    if(!this->MultipoleOrder()) return;
    int np=1;
    if(setup_data.interac_data.Dim(0)==0 || setup_data.interac_data.Dim(1)==0){
      return;
    }
    Profile::Tic("Host2Device",false,25);
    int level=setup_data.level;
    size_t buff_size=*((size_t*)&setup_data.interac_data[0][0]);
    typename Vector<char>::Device          buff;
    typename Matrix<char>::Device  interac_data;
    typename Matrix<Real_t>::Device  input_data;
    typename Matrix<Real_t>::Device output_data;
    if(this->dev_buffer.Dim()<buff_size) this->dev_buffer.ReInit(buff_size);
    buff        =       this-> dev_buffer;
    interac_data= setup_data.interac_data;
    input_data  =*setup_data.  input_data;
    output_data =*setup_data. output_data;
    Profile::Toc();
    {
      size_t m, dof, ker_dim0, ker_dim1, n_blk0;
      std::vector<Vector<size_t> >  fft_vec;
      std::vector<Vector<size_t> > ifft_vec;
      std::vector<Vector<Real_t> >  fft_scl;
      std::vector<Vector<Real_t> > ifft_scl;
      std::vector<Vector<size_t> > interac_vec;
      std::vector<Vector<size_t> > interac_dsp;
      Vector<Real_t*> precomp_mat;
      {
        char* data_ptr=&interac_data[0][0];
        buff_size=((size_t*)data_ptr)[0]; data_ptr+=sizeof(size_t);
        m        =((size_t*)data_ptr)[0]; data_ptr+=sizeof(size_t);
        dof      =((size_t*)data_ptr)[0]; data_ptr+=sizeof(size_t);
        ker_dim0 =((size_t*)data_ptr)[0]; data_ptr+=sizeof(size_t);
        ker_dim1 =((size_t*)data_ptr)[0]; data_ptr+=sizeof(size_t);
        n_blk0   =((size_t*)data_ptr)[0]; data_ptr+=sizeof(size_t);
        fft_vec .resize(n_blk0);
        ifft_vec.resize(n_blk0);
        fft_scl .resize(n_blk0);
        ifft_scl.resize(n_blk0);
        interac_vec.resize(n_blk0);
        interac_dsp.resize(n_blk0);
        Vector<size_t> interac_mat;
        interac_mat.ReInit(((size_t*)data_ptr)[0],(size_t*)(data_ptr+sizeof(size_t)),false);
        data_ptr+=sizeof(size_t)+interac_mat.Dim()*sizeof(size_t);
        Vector<Real_t*> interac_mat_ptr;
        interac_mat_ptr.ReInit(((size_t*)data_ptr)[0],(Real_t**)(data_ptr+sizeof(size_t)),false);
        data_ptr+=sizeof(size_t)+interac_mat_ptr.Dim()*sizeof(Real_t*);
        precomp_mat.Resize(interac_mat_ptr.Dim());
        for(size_t i=0;i<interac_mat_ptr.Dim();i++){
          precomp_mat[i]=interac_mat_ptr[i];
        }
        for(size_t blk0=0;blk0<n_blk0;blk0++){
          fft_vec[blk0].ReInit(((size_t*)data_ptr)[0],(size_t*)(data_ptr+sizeof(size_t)),false);
          data_ptr+=sizeof(size_t)+fft_vec[blk0].Dim()*sizeof(size_t);
          ifft_vec[blk0].ReInit(((size_t*)data_ptr)[0],(size_t*)(data_ptr+sizeof(size_t)),false);
          data_ptr+=sizeof(size_t)+ifft_vec[blk0].Dim()*sizeof(size_t);
          fft_scl[blk0].ReInit(((size_t*)data_ptr)[0],(Real_t*)(data_ptr+sizeof(size_t)),false);
          data_ptr+=sizeof(size_t)+fft_scl[blk0].Dim()*sizeof(Real_t);
          ifft_scl[blk0].ReInit(((size_t*)data_ptr)[0],(Real_t*)(data_ptr+sizeof(size_t)),false);
          data_ptr+=sizeof(size_t)+ifft_scl[blk0].Dim()*sizeof(Real_t);
          interac_vec[blk0].ReInit(((size_t*)data_ptr)[0],(size_t*)(data_ptr+sizeof(size_t)),false);
          data_ptr+=sizeof(size_t)+interac_vec[blk0].Dim()*sizeof(size_t);
          interac_dsp[blk0].ReInit(((size_t*)data_ptr)[0],(size_t*)(data_ptr+sizeof(size_t)),false);
          data_ptr+=sizeof(size_t)+interac_dsp[blk0].Dim()*sizeof(size_t);
        }
      }
      int omp_p=omp_get_max_threads();
      size_t M_dim, fftsize;
      {
        size_t n1=m*2;
        size_t n2=n1*n1;
        size_t n3_=n2*(n1/2+1);
        size_t chld_cnt=1UL<<COORD_DIM;
        fftsize=2*n3_*chld_cnt;
        M_dim=n3_;
      }
      for(size_t blk0=0;blk0<n_blk0;blk0++){
        size_t n_in = fft_vec[blk0].Dim();
        size_t n_out=ifft_vec[blk0].Dim();
        size_t  input_dim=n_in *ker_dim0*dof*fftsize;
        size_t output_dim=n_out*ker_dim1*dof*fftsize;
        size_t buffer_dim=2*(ker_dim0+ker_dim1)*dof*fftsize*omp_p;
        Vector<Real_t> fft_in ( input_dim, (Real_t*)&buff[         0                           ],false);
        Vector<Real_t> fft_out(output_dim, (Real_t*)&buff[ input_dim            *sizeof(Real_t)],false);
        Vector<Real_t>  buffer(buffer_dim, (Real_t*)&buff[(input_dim+output_dim)*sizeof(Real_t)],false);
        {
          if(np==1) Profile::Tic("FFT",false,100);
          Vector<Real_t>  input_data_( input_data.dim[0]* input_data.dim[1],  input_data[0], false);
          FFT_UpEquiv(dof, m, ker_dim0,  fft_vec[blk0],  fft_scl[blk0],  input_data_, fft_in, buffer);
          if(np==1) Profile::Toc();
        }
        {
#ifdef PVFMM_HAVE_PAPI
#ifdef __VERBOSE__
          std::cout << "Starting counters new\n";
          if (PAPI_start(EventSet) != PAPI_OK) std::cout << "handle_error3" << std::endl;
#endif
#endif
          if(np==1) Profile::Tic("HadamardProduct",false,100);
          VListHadamard<Real_t>(dof, M_dim, ker_dim0, ker_dim1, interac_dsp[blk0], interac_vec[blk0], precomp_mat, fft_in, fft_out);
          if(np==1) Profile::Toc();
#ifdef PVFMM_HAVE_PAPI
#ifdef __VERBOSE__
          if (PAPI_stop(EventSet, values) != PAPI_OK) std::cout << "handle_error4" << std::endl;
          std::cout << "Stopping counters\n";
#endif
#endif
        }
        {
          if(np==1) Profile::Tic("IFFT",false,100);
          Vector<Real_t> output_data_(output_data.dim[0]*output_data.dim[1], output_data[0], false);
          FFT_Check2Equiv(dof, m, ker_dim1, ifft_vec[blk0], ifft_scl[blk0], fft_out, output_data_, buffer);
          if(np==1) Profile::Toc();
        }
      }
    }
  }

  void Down2DownSetup(SetupData<Real_t,FMMNode_t>& setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level){
    if(!this->MultipoleOrder()) return;
    {
      setup_data.level=level;
      setup_data.kernel=kernel->k_l2l;
      setup_data.interac_type.resize(1);
      setup_data.interac_type[0]=D2D_Type;
      setup_data. input_data=&buff[1];
      setup_data.output_data=&buff[1];
      Vector<FMMNode_t*>& nodes_in =n_list[1];
      Vector<FMMNode_t*>& nodes_out=n_list[1];
      setup_data.nodes_in .clear();
      setup_data.nodes_out.clear();
      for(size_t i=0;i<nodes_in .Dim();i++) if((nodes_in [i]->depth==level-1) && nodes_in [i]->pt_cnt[1]) setup_data.nodes_in .push_back(nodes_in [i]);
      for(size_t i=0;i<nodes_out.Dim();i++) if((nodes_out[i]->depth==level  ) && nodes_out[i]->pt_cnt[1]) setup_data.nodes_out.push_back(nodes_out[i]);
    }
    std::vector<FMMNode_t*>& nodes_in =setup_data.nodes_in ;
    std::vector<FMMNode_t*>& nodes_out=setup_data.nodes_out;
    std::vector<Vector<Real_t>*>&  input_vector=setup_data. input_vector;  input_vector.clear();
    std::vector<Vector<Real_t>*>& output_vector=setup_data.output_vector; output_vector.clear();
    for(size_t i=0;i<nodes_in .size();i++)  input_vector.push_back(&(nodes_in[i]->FMMData())->dnward_equiv);
    for(size_t i=0;i<nodes_out.size();i++) output_vector.push_back(&(nodes_out[i]->FMMData())->dnward_equiv);
    SetupInterac(setup_data);
  }
  
  void Down2Down(SetupData<Real_t,FMMNode_t>& setup_data){
    if(!this->MultipoleOrder()) return;
    EvalList(setup_data);
  }

  void X_ListSetup(SetupData<Real_t,FMMNode_t>&  setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level){
    if(!this->MultipoleOrder()) return;
    {
      setup_data. level=level;
      setup_data.kernel=kernel->k_s2l;
      setup_data. input_data=&buff[4];
      setup_data.output_data=&buff[1];
      setup_data. coord_data=&buff[6];
      Vector<FMMNode_t*>& nodes_in =n_list[4];
      Vector<FMMNode_t*>& nodes_out=n_list[1];
      setup_data.nodes_in .clear();
      setup_data.nodes_out.clear();
      for(size_t i=0;i<nodes_in .Dim();i++) if((level==0 || level==-1) && (nodes_in [i]->src_coord.Dim() || nodes_in [i]->surf_coord.Dim()) &&  nodes_in [i]->IsLeaf ()) setup_data.nodes_in .push_back(nodes_in [i]);
      for(size_t i=0;i<nodes_out.Dim();i++) if((level==0 || level==-1) &&  nodes_out[i]->pt_cnt[1]                                          && !nodes_out[i]->IsGhost()) setup_data.nodes_out.push_back(nodes_out[i]);
    }
    ptSetupData data;
    data. level=setup_data. level;
    data.kernel=setup_data.kernel;
    std::vector<FMMNode_t*>& nodes_in =setup_data.nodes_in ;
    std::vector<FMMNode_t*>& nodes_out=setup_data.nodes_out;
    {
      std::vector<FMMNode_t*>& nodes=nodes_in;
      PackedData& coord=data.src_coord;
      PackedData& value=data.src_value;
      coord.ptr=setup_data. coord_data;
      value.ptr=setup_data. input_data;
      coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
      value.len=value.ptr->Dim(0)*value.ptr->Dim(1);
      coord.cnt.ReInit(nodes.size());
      coord.dsp.ReInit(nodes.size());
      value.cnt.ReInit(nodes.size());
      value.dsp.ReInit(nodes.size());
#pragma omp parallel for
      for(size_t i=0;i<nodes.size();i++){
        ((FMMNode_t*)nodes[i])->node_id=i;
        Vector<Real_t>& coord_vec=nodes[i]->src_coord;
        Vector<Real_t>& value_vec=nodes[i]->src_value;
        if(coord_vec.Dim()){
          coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
          assert(coord.dsp[i]<coord.len);
          coord.cnt[i]=coord_vec.Dim();
        }else{
          coord.dsp[i]=0;
          coord.cnt[i]=0;
        }
        if(value_vec.Dim()){
          value.dsp[i]=&value_vec[0]-value.ptr[0][0];
          assert(value.dsp[i]<value.len);
          value.cnt[i]=value_vec.Dim();
        }else{
          value.dsp[i]=0;
          value.cnt[i]=0;
        }
      }
    }
    {
      std::vector<FMMNode_t*>& nodes=nodes_in;
      PackedData& coord=data.srf_coord;
      PackedData& value=data.srf_value;
      coord.ptr=setup_data. coord_data;
      value.ptr=setup_data. input_data;
      coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
      value.len=value.ptr->Dim(0)*value.ptr->Dim(1);
      coord.cnt.ReInit(nodes.size());
      coord.dsp.ReInit(nodes.size());
      value.cnt.ReInit(nodes.size());
      value.dsp.ReInit(nodes.size());
#pragma omp parallel for
      for(size_t i=0;i<nodes.size();i++){
        Vector<Real_t>& coord_vec=nodes[i]->surf_coord;
        Vector<Real_t>& value_vec=nodes[i]->surf_value;
        if(coord_vec.Dim()){
          coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
          assert(coord.dsp[i]<coord.len);
          coord.cnt[i]=coord_vec.Dim();
        }else{
          coord.dsp[i]=0;
          coord.cnt[i]=0;
        }
        if(value_vec.Dim()){
          value.dsp[i]=&value_vec[0]-value.ptr[0][0];
          assert(value.dsp[i]<value.len);
          value.cnt[i]=value_vec.Dim();
        }else{
          value.dsp[i]=0;
          value.cnt[i]=0;
        }
      }
    }
    {
      std::vector<FMMNode_t*>& nodes=nodes_out;
      PackedData& coord=data.trg_coord;
      PackedData& value=data.trg_value;
      coord.ptr=setup_data. coord_data;
      value.ptr=setup_data.output_data;
      coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
      value.len=value.ptr->Dim(0)*value.ptr->Dim(1);
      coord.cnt.ReInit(nodes.size());
      coord.dsp.ReInit(nodes.size());
      value.cnt.ReInit(nodes.size());
      value.dsp.ReInit(nodes.size());
#pragma omp parallel for
      for(size_t i=0;i<nodes.size();i++){
        Vector<Real_t>& coord_vec=tree->dnwd_check_surf[nodes[i]->depth];
        Vector<Real_t>& value_vec=(nodes[i]->FMMData())->dnward_equiv;
        if(coord_vec.Dim()){
          coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
          assert(coord.dsp[i]<coord.len);
          coord.cnt[i]=coord_vec.Dim();
        }else{
          coord.dsp[i]=0;
          coord.cnt[i]=0;
        }
        if(value_vec.Dim()){
          value.dsp[i]=&value_vec[0]-value.ptr[0][0];
          assert(value.dsp[i]<value.len);
          value.cnt[i]=value_vec.Dim();
        }else{
          value.dsp[i]=0;
          value.cnt[i]=0;
        }
      }
    }
    {
      int omp_p=omp_get_max_threads();
      std::vector<std::vector<size_t> > in_node_(omp_p);
      std::vector<std::vector<size_t> > scal_idx_(omp_p);
      std::vector<std::vector<Real_t> > coord_shift_(omp_p);
      std::vector<std::vector<size_t> > interac_cnt_(omp_p);
      size_t m=this->MultipoleOrder();
      size_t Nsrf=(6*(m-1)*(m-1)+2);
#pragma omp parallel for
      for(size_t tid=0;tid<omp_p;tid++){
        std::vector<size_t>& in_node    =in_node_[tid];
        std::vector<size_t>& scal_idx   =scal_idx_[tid];
        std::vector<Real_t>& coord_shift=coord_shift_[tid];
        std::vector<size_t>& interac_cnt=interac_cnt_[tid];
        size_t a=(nodes_out.size()*(tid+0))/omp_p;
        size_t b=(nodes_out.size()*(tid+1))/omp_p;
        for(size_t i=a;i<b;i++){
          FMMNode_t* tnode=nodes_out[i];
          if(tnode->IsLeaf() && tnode->pt_cnt[1]<=Nsrf){
            interac_cnt.push_back(0);
            continue;
          }
          Real_t s=pvfmm::pow<Real_t>(0.5,tnode->depth);
          size_t interac_cnt_=0;
          {
            Mat_Type type=X_Type;
            Vector<FMMNode_t*>& intlst=tnode->interac_list[type];
            for(size_t j=0;j<intlst.Dim();j++) if(intlst[j]){
              FMMNode_t* snode=intlst[j];
              size_t snode_id=snode->node_id;
              if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
              in_node.push_back(snode_id);
              scal_idx.push_back(snode->depth);
              {
                const int* rel_coord=interac_list.RelativeCoord(type,j);
                const Real_t* scoord=snode->Coord();
                const Real_t* tcoord=tnode->Coord();
                Real_t shift[COORD_DIM];
                shift[0]=rel_coord[0]*0.5*s-(scoord[0]+1.0*s)+(0+0.5*s);
                shift[1]=rel_coord[1]*0.5*s-(scoord[1]+1.0*s)+(0+0.5*s);
                shift[2]=rel_coord[2]*0.5*s-(scoord[2]+1.0*s)+(0+0.5*s);
                coord_shift.push_back(shift[0]);
                coord_shift.push_back(shift[1]);
                coord_shift.push_back(shift[2]);
              }
              interac_cnt_++;
            }
          }
          interac_cnt.push_back(interac_cnt_);
        }
      }
      {
        InteracData& interac_data=data.interac_data;
	CopyVec(in_node_,interac_data.in_node);
	CopyVec(scal_idx_,interac_data.scal_idx);
	CopyVec(coord_shift_,interac_data.coord_shift);
	CopyVec(interac_cnt_,interac_data.interac_cnt);
        {
          pvfmm::Vector<size_t>& cnt=interac_data.interac_cnt;
          pvfmm::Vector<size_t>& dsp=interac_data.interac_dsp;
          dsp.ReInit(cnt.Dim()); if(dsp.Dim()) dsp[0]=0;
          omp_par::scan(&cnt[0],&dsp[0],dsp.Dim());
        }
      }
    }
    PtSetup(setup_data, &data);
  }
  
  void X_List(SetupData<Real_t,FMMNode_t>&  setup_data){
    if(!this->MultipoleOrder()) return;
    this->EvalListPts(setup_data);
  }
  
  void W_ListSetup(SetupData<Real_t,FMMNode_t>&  setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level){
    if(!this->MultipoleOrder()) return;
    {
      setup_data. level=level;
      setup_data.kernel=kernel->k_m2t;
      setup_data. input_data=&buff[0];
      setup_data.output_data=&buff[5];
      setup_data. coord_data=&buff[6];
      Vector<FMMNode_t*>& nodes_in =n_list[0];
      Vector<FMMNode_t*>& nodes_out=n_list[5];
      setup_data.nodes_in .clear();
      setup_data.nodes_out.clear();
      for(size_t i=0;i<nodes_in .Dim();i++) if((level==0 || level==-1) && nodes_in [i]->pt_cnt[0]                                                            ) setup_data.nodes_in .push_back(nodes_in [i]);
      for(size_t i=0;i<nodes_out.Dim();i++) if((level==0 || level==-1) && nodes_out[i]->trg_coord.Dim() && nodes_out[i]->IsLeaf() && !nodes_out[i]->IsGhost()) setup_data.nodes_out.push_back(nodes_out[i]);
    }
    ptSetupData data;
    data. level=setup_data. level;
    data.kernel=setup_data.kernel;
    std::vector<FMMNode_t*>& nodes_in =setup_data.nodes_in ;
    std::vector<FMMNode_t*>& nodes_out=setup_data.nodes_out;
    {
      std::vector<FMMNode_t*>& nodes=nodes_in;
      PackedData& coord=data.src_coord;
      PackedData& value=data.src_value;
      coord.ptr=setup_data. coord_data;
      value.ptr=setup_data. input_data;
      coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
      value.len=value.ptr->Dim(0)*value.ptr->Dim(1);
      coord.cnt.ReInit(nodes.size());
      coord.dsp.ReInit(nodes.size());
      value.cnt.ReInit(nodes.size());
      value.dsp.ReInit(nodes.size());
#pragma omp parallel for
      for(size_t i=0;i<nodes.size();i++){
        ((FMMNode_t*)nodes[i])->node_id=i;
        Vector<Real_t>& coord_vec=tree->upwd_equiv_surf[nodes[i]->depth];
        Vector<Real_t>& value_vec=(nodes[i]->FMMData())->upward_equiv;
        if(coord_vec.Dim()){
          coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
          assert(coord.dsp[i]<coord.len);
          coord.cnt[i]=coord_vec.Dim();
        }else{
          coord.dsp[i]=0;
          coord.cnt[i]=0;
        }
        if(value_vec.Dim()){
          value.dsp[i]=&value_vec[0]-value.ptr[0][0];
          assert(value.dsp[i]<value.len);
          value.cnt[i]=value_vec.Dim();
        }else{
          value.dsp[i]=0;
          value.cnt[i]=0;
        }
      }
    }
    {
      std::vector<FMMNode_t*>& nodes=nodes_in;
      PackedData& coord=data.srf_coord;
      PackedData& value=data.srf_value;
      coord.ptr=setup_data. coord_data;
      value.ptr=setup_data. input_data;
      coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
      value.len=value.ptr->Dim(0)*value.ptr->Dim(1);
      coord.cnt.ReInit(nodes.size());
      coord.dsp.ReInit(nodes.size());
      value.cnt.ReInit(nodes.size());
      value.dsp.ReInit(nodes.size());
#pragma omp parallel for
      for(size_t i=0;i<nodes.size();i++){
        coord.dsp[i]=0;
        coord.cnt[i]=0;
        value.dsp[i]=0;
        value.cnt[i]=0;
      }
    }
    {
      std::vector<FMMNode_t*>& nodes=nodes_out;
      PackedData& coord=data.trg_coord;
      PackedData& value=data.trg_value;
      coord.ptr=setup_data. coord_data;
      value.ptr=setup_data.output_data;
      coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
      value.len=value.ptr->Dim(0)*value.ptr->Dim(1);
      coord.cnt.ReInit(nodes.size());
      coord.dsp.ReInit(nodes.size());
      value.cnt.ReInit(nodes.size());
      value.dsp.ReInit(nodes.size());
#pragma omp parallel for
      for(size_t i=0;i<nodes.size();i++){
        Vector<Real_t>& coord_vec=nodes[i]->trg_coord;
        Vector<Real_t>& value_vec=nodes[i]->trg_value;
        if(coord_vec.Dim()){
          coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
          assert(coord.dsp[i]<coord.len);
          coord.cnt[i]=coord_vec.Dim();
        }else{
          coord.dsp[i]=0;
          coord.cnt[i]=0;
        }
        if(value_vec.Dim()){
          value.dsp[i]=&value_vec[0]-value.ptr[0][0];
          assert(value.dsp[i]<value.len);
          value.cnt[i]=value_vec.Dim();
        }else{
          value.dsp[i]=0;
          value.cnt[i]=0;
        }
      }
    }
    {
      int omp_p=omp_get_max_threads();
      std::vector<std::vector<size_t> > in_node_(omp_p);
      std::vector<std::vector<size_t> > scal_idx_(omp_p);
      std::vector<std::vector<Real_t> > coord_shift_(omp_p);
      std::vector<std::vector<size_t> > interac_cnt_(omp_p);
      size_t m=this->MultipoleOrder();
      size_t Nsrf=(6*(m-1)*(m-1)+2);
#pragma omp parallel for
      for(size_t tid=0;tid<omp_p;tid++){
        std::vector<size_t>& in_node    =in_node_[tid]    ;
        std::vector<size_t>& scal_idx   =scal_idx_[tid]   ;
        std::vector<Real_t>& coord_shift=coord_shift_[tid];
        std::vector<size_t>& interac_cnt=interac_cnt_[tid]        ;
        size_t a=(nodes_out.size()*(tid+0))/omp_p;
        size_t b=(nodes_out.size()*(tid+1))/omp_p;
        for(size_t i=a;i<b;i++){
          FMMNode_t* tnode=nodes_out[i];
          Real_t s=pvfmm::pow<Real_t>(0.5,tnode->depth);
          size_t interac_cnt_=0;
          {
            Mat_Type type=W_Type;
            Vector<FMMNode_t*>& intlst=tnode->interac_list[type];
            for(size_t j=0;j<intlst.Dim();j++) if(intlst[j]){
              FMMNode_t* snode=intlst[j];
              size_t snode_id=snode->node_id;
              if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
              if(snode->IsGhost() && snode->src_coord.Dim()+snode->surf_coord.Dim()==0){
              }else if(snode->IsLeaf() && snode->pt_cnt[0]<=Nsrf) continue;
              in_node.push_back(snode_id);
              scal_idx.push_back(snode->depth);
              {
                const int* rel_coord=interac_list.RelativeCoord(type,j);
                const Real_t* scoord=snode->Coord();
                const Real_t* tcoord=tnode->Coord();
                Real_t shift[COORD_DIM];
                shift[0]=rel_coord[0]*0.25*s-(0+0.25*s)+(tcoord[0]+0.5*s);
                shift[1]=rel_coord[1]*0.25*s-(0+0.25*s)+(tcoord[1]+0.5*s);
                shift[2]=rel_coord[2]*0.25*s-(0+0.25*s)+(tcoord[2]+0.5*s);
                coord_shift.push_back(shift[0]);
                coord_shift.push_back(shift[1]);
                coord_shift.push_back(shift[2]);
              }
              interac_cnt_++;
            }
          }
          interac_cnt.push_back(interac_cnt_);
        }
      }
      {
        InteracData& interac_data=data.interac_data;
	CopyVec(in_node_,interac_data.in_node);
	CopyVec(scal_idx_,interac_data.scal_idx);
	CopyVec(coord_shift_,interac_data.coord_shift);
	CopyVec(interac_cnt_,interac_data.interac_cnt);
        {
          pvfmm::Vector<size_t>& cnt=interac_data.interac_cnt;
          pvfmm::Vector<size_t>& dsp=interac_data.interac_dsp;
          dsp.ReInit(cnt.Dim()); if(dsp.Dim()) dsp[0]=0;
          omp_par::scan(&cnt[0],&dsp[0],dsp.Dim());
        }
      }
    }
    PtSetup(setup_data, &data);
  }
  
  void W_List(SetupData<Real_t,FMMNode_t>&  setup_data){
    if(!this->MultipoleOrder()) return;
    this->EvalListPts(setup_data);
  }  

  void U_ListSetup(SetupData<Real_t,FMMNode_t>& setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level){
    {
      setup_data. level=level;
      setup_data.kernel=kernel->k_s2t;
      setup_data. input_data=&buff[4];
      setup_data.output_data=&buff[5];
      setup_data. coord_data=&buff[6];
      Vector<FMMNode_t*>& nodes_in =n_list[4];
      Vector<FMMNode_t*>& nodes_out=n_list[5];
      setup_data.nodes_in .clear();
      setup_data.nodes_out.clear();
      for(size_t i=0;i<nodes_in .Dim();i++)
        if((level==0 || level==-1)
  	 && (nodes_in [i]->src_coord.Dim() || nodes_in [i]->surf_coord.Dim())
  	 && nodes_in [i]->IsLeaf()                            ) setup_data.nodes_in .push_back(nodes_in [i]);
      for(size_t i=0;i<nodes_out.Dim();i++)
        if((level==0 || level==-1)
  	 && (nodes_out[i]->trg_coord.Dim()                                  )
  	 && nodes_out[i]->IsLeaf() && !nodes_out[i]->IsGhost()) setup_data.nodes_out.push_back(nodes_out[i]);
    }
    ptSetupData data;
    data. level=setup_data. level;
    data.kernel=setup_data.kernel;
    std::vector<FMMNode_t*>& nodes_in =setup_data.nodes_in ;
    std::vector<FMMNode_t*>& nodes_out=setup_data.nodes_out;
    {
      std::vector<FMMNode_t*>& nodes=nodes_in;
      PackedData& coord=data.src_coord;
      PackedData& value=data.src_value;
      coord.ptr=setup_data. coord_data;
      value.ptr=setup_data. input_data;
      coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
      value.len=value.ptr->Dim(0)*value.ptr->Dim(1);
      coord.cnt.ReInit(nodes.size());
      coord.dsp.ReInit(nodes.size());
      value.cnt.ReInit(nodes.size());
      value.dsp.ReInit(nodes.size());
#pragma omp parallel for
      for(size_t i=0;i<nodes.size();i++){
        nodes[i]->node_id=i;
        Vector<Real_t>& coord_vec=nodes[i]->src_coord;
        Vector<Real_t>& value_vec=nodes[i]->src_value;
        if(coord_vec.Dim()){
          coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
          assert(coord.dsp[i]<coord.len);
          coord.cnt[i]=coord_vec.Dim();
        }else{
          coord.dsp[i]=0;
          coord.cnt[i]=0;
        }
        if(value_vec.Dim()){
          value.dsp[i]=&value_vec[0]-value.ptr[0][0];
          assert(value.dsp[i]<value.len);
          value.cnt[i]=value_vec.Dim();
        }else{
          value.dsp[i]=0;
          value.cnt[i]=0;
        }
      }
    }
    {
      std::vector<FMMNode_t*>& nodes=nodes_in;
      PackedData& coord=data.srf_coord;
      PackedData& value=data.srf_value;
      coord.ptr=setup_data. coord_data;
      value.ptr=setup_data. input_data;
      coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
      value.len=value.ptr->Dim(0)*value.ptr->Dim(1);
      coord.cnt.ReInit(nodes.size());
      coord.dsp.ReInit(nodes.size());
      value.cnt.ReInit(nodes.size());
      value.dsp.ReInit(nodes.size());
#pragma omp parallel for
      for(size_t i=0;i<nodes.size();i++){
        Vector<Real_t>& coord_vec=nodes[i]->surf_coord;
        Vector<Real_t>& value_vec=nodes[i]->surf_value;
        if(coord_vec.Dim()){
          coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
          assert(coord.dsp[i]<coord.len);
          coord.cnt[i]=coord_vec.Dim();
        }else{
          coord.dsp[i]=0;
          coord.cnt[i]=0;
        }
        if(value_vec.Dim()){
          value.dsp[i]=&value_vec[0]-value.ptr[0][0];
          assert(value.dsp[i]<value.len);
          value.cnt[i]=value_vec.Dim();
        }else{
          value.dsp[i]=0;
          value.cnt[i]=0;
        }
      }
    }
    {
      std::vector<FMMNode_t*>& nodes=nodes_out;
      PackedData& coord=data.trg_coord;
      PackedData& value=data.trg_value;
      coord.ptr=setup_data. coord_data;
      value.ptr=setup_data.output_data;
      coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
      value.len=value.ptr->Dim(0)*value.ptr->Dim(1);
      coord.cnt.ReInit(nodes.size());
      coord.dsp.ReInit(nodes.size());
      value.cnt.ReInit(nodes.size());
      value.dsp.ReInit(nodes.size());
#pragma omp parallel for
      for(size_t i=0;i<nodes.size();i++){
        Vector<Real_t>& coord_vec=nodes[i]->trg_coord;
        Vector<Real_t>& value_vec=nodes[i]->trg_value;
        if(coord_vec.Dim()){
          coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
          assert(coord.dsp[i]<coord.len);
          coord.cnt[i]=coord_vec.Dim();
        }else{
          coord.dsp[i]=0;
          coord.cnt[i]=0;
        }
        if(value_vec.Dim()){
          value.dsp[i]=&value_vec[0]-value.ptr[0][0];
          assert(value.dsp[i]<value.len);
          value.cnt[i]=value_vec.Dim();
        }else{
          value.dsp[i]=0;
          value.cnt[i]=0;
        }
      }
    }
    {
      int omp_p=omp_get_max_threads();
      std::vector<std::vector<size_t> > in_node_(omp_p);
      std::vector<std::vector<size_t> > scal_idx_(omp_p);
      std::vector<std::vector<Real_t> > coord_shift_(omp_p);
      std::vector<std::vector<size_t> > interac_cnt_(omp_p);
      size_t m=this->MultipoleOrder();
      size_t Nsrf=(6*(m-1)*(m-1)+2);
#pragma omp parallel for
      for(size_t tid=0;tid<omp_p;tid++){
        std::vector<size_t>& in_node    =in_node_[tid]    ;
        std::vector<size_t>& scal_idx   =scal_idx_[tid]   ;
        std::vector<Real_t>& coord_shift=coord_shift_[tid];
        std::vector<size_t>& interac_cnt=interac_cnt_[tid]        ;
        size_t a=(nodes_out.size()*(tid+0))/omp_p;
        size_t b=(nodes_out.size()*(tid+1))/omp_p;
        for(size_t i=a;i<b;i++){
          FMMNode_t* tnode=nodes_out[i];
          Real_t s=pvfmm::pow<Real_t>(0.5,tnode->depth);
          size_t interac_cnt_=0;
          {
            Mat_Type type=U0_Type;
            Vector<FMMNode_t*>& intlst=tnode->interac_list[type];
            for(size_t j=0;j<intlst.Dim();j++) if(intlst[j]){
              FMMNode_t* snode=intlst[j];
              size_t snode_id=snode->node_id;
              if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
              in_node.push_back(snode_id);
              scal_idx.push_back(snode->depth);
              {
                const int* rel_coord=interac_list.RelativeCoord(type,j);
                const Real_t* scoord=snode->Coord();
                const Real_t* tcoord=tnode->Coord();
                Real_t shift[COORD_DIM];
                shift[0]=rel_coord[0]*0.5*s-(scoord[0]+1.0*s)+(tcoord[0]+0.5*s);
                shift[1]=rel_coord[1]*0.5*s-(scoord[1]+1.0*s)+(tcoord[1]+0.5*s);
                shift[2]=rel_coord[2]*0.5*s-(scoord[2]+1.0*s)+(tcoord[2]+0.5*s);
                coord_shift.push_back(shift[0]);
                coord_shift.push_back(shift[1]);
                coord_shift.push_back(shift[2]);
              }
              interac_cnt_++;
            }
          }
          {
            Mat_Type type=U1_Type;
            Vector<FMMNode_t*>& intlst=tnode->interac_list[type];
            for(size_t j=0;j<intlst.Dim();j++) if(intlst[j]){
              FMMNode_t* snode=intlst[j];
              size_t snode_id=snode->node_id;
              if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
              in_node.push_back(snode_id);
              scal_idx.push_back(snode->depth);
              {
                const int* rel_coord=interac_list.RelativeCoord(type,j);
                const Real_t* scoord=snode->Coord();
                const Real_t* tcoord=tnode->Coord();
                Real_t shift[COORD_DIM];
                shift[0]=rel_coord[0]*1.0*s-(scoord[0]+0.5*s)+(tcoord[0]+0.5*s);
                shift[1]=rel_coord[1]*1.0*s-(scoord[1]+0.5*s)+(tcoord[1]+0.5*s);
                shift[2]=rel_coord[2]*1.0*s-(scoord[2]+0.5*s)+(tcoord[2]+0.5*s);
                coord_shift.push_back(shift[0]);
                coord_shift.push_back(shift[1]);
                coord_shift.push_back(shift[2]);
              }
              interac_cnt_++;
            }
          }
          {
            Mat_Type type=U2_Type;
            Vector<FMMNode_t*>& intlst=tnode->interac_list[type];
            for(size_t j=0;j<intlst.Dim();j++) if(intlst[j]){
              FMMNode_t* snode=intlst[j];
              size_t snode_id=snode->node_id;
              if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
              in_node.push_back(snode_id);
              scal_idx.push_back(snode->depth);
              {
                const int* rel_coord=interac_list.RelativeCoord(type,j);
                const Real_t* scoord=snode->Coord();
                const Real_t* tcoord=tnode->Coord();
                Real_t shift[COORD_DIM];
                shift[0]=rel_coord[0]*0.25*s-(scoord[0]+0.25*s)+(tcoord[0]+0.5*s);
                shift[1]=rel_coord[1]*0.25*s-(scoord[1]+0.25*s)+(tcoord[1]+0.5*s);
                shift[2]=rel_coord[2]*0.25*s-(scoord[2]+0.25*s)+(tcoord[2]+0.5*s);
                coord_shift.push_back(shift[0]);
                coord_shift.push_back(shift[1]);
                coord_shift.push_back(shift[2]);
              }
              interac_cnt_++;
            }
          }
          {
            Mat_Type type=X_Type;
            Vector<FMMNode_t*>& intlst=tnode->interac_list[type];
            if(tnode->pt_cnt[1]<=Nsrf)
            for(size_t j=0;j<intlst.Dim();j++) if(intlst[j]){
              FMMNode_t* snode=intlst[j];
              size_t snode_id=snode->node_id;
              if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
              in_node.push_back(snode_id);
              scal_idx.push_back(snode->depth);
              {
                const int* rel_coord=interac_list.RelativeCoord(type,j);
                const Real_t* scoord=snode->Coord();
                const Real_t* tcoord=tnode->Coord();
                Real_t shift[COORD_DIM];
                shift[0]=rel_coord[0]*0.5*s-(scoord[0]+1.0*s)+(tcoord[0]+0.5*s);
                shift[1]=rel_coord[1]*0.5*s-(scoord[1]+1.0*s)+(tcoord[1]+0.5*s);
                shift[2]=rel_coord[2]*0.5*s-(scoord[2]+1.0*s)+(tcoord[2]+0.5*s);
                coord_shift.push_back(shift[0]);
                coord_shift.push_back(shift[1]);
                coord_shift.push_back(shift[2]);
              }
              interac_cnt_++;
            }
          }
          {
            Mat_Type type=W_Type;
            Vector<FMMNode_t*>& intlst=tnode->interac_list[type];
            for(size_t j=0;j<intlst.Dim();j++) if(intlst[j]){
              FMMNode_t* snode=intlst[j];
              size_t snode_id=snode->node_id;
              if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
              if(snode->IsGhost() && snode->src_coord.Dim()+snode->surf_coord.Dim()==0) continue;
              if(snode->pt_cnt[0]> Nsrf) continue;
              in_node.push_back(snode_id);
              scal_idx.push_back(snode->depth);
              {
                const int* rel_coord=interac_list.RelativeCoord(type,j);
                const Real_t* scoord=snode->Coord();
                const Real_t* tcoord=tnode->Coord();
                Real_t shift[COORD_DIM];
                shift[0]=rel_coord[0]*0.25*s-(scoord[0]+0.25*s)+(tcoord[0]+0.5*s);
                shift[1]=rel_coord[1]*0.25*s-(scoord[1]+0.25*s)+(tcoord[1]+0.5*s);
                shift[2]=rel_coord[2]*0.25*s-(scoord[2]+0.25*s)+(tcoord[2]+0.5*s);
                coord_shift.push_back(shift[0]);
                coord_shift.push_back(shift[1]);
                coord_shift.push_back(shift[2]);
              }
              interac_cnt_++;
            }
          }
          interac_cnt.push_back(interac_cnt_);
        }
      }
      {
        InteracData& interac_data=data.interac_data;
	CopyVec(in_node_,interac_data.in_node);
	CopyVec(scal_idx_,interac_data.scal_idx);
	CopyVec(coord_shift_,interac_data.coord_shift);
	CopyVec(interac_cnt_,interac_data.interac_cnt);
        {
          pvfmm::Vector<size_t>& cnt=interac_data.interac_cnt;
          pvfmm::Vector<size_t>& dsp=interac_data.interac_dsp;
          dsp.ReInit(cnt.Dim()); if(dsp.Dim()) dsp[0]=0;
          omp_par::scan(&cnt[0],&dsp[0],dsp.Dim());
        }
      }
    }
    PtSetup(setup_data, &data);
  }

  void U_List(SetupData<Real_t,FMMNode_t>&  setup_data){
    this->EvalListPts(setup_data);
  }
  
  void Down2TargetSetup(SetupData<Real_t,FMMNode_t>&  setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level){
    if(!this->MultipoleOrder()) return;
    {
      setup_data. level=level;
      setup_data.kernel=kernel->k_l2t;
      setup_data. input_data=&buff[1];
      setup_data.output_data=&buff[5];
      setup_data. coord_data=&buff[6];
      Vector<FMMNode_t*>& nodes_in =n_list[1];
      Vector<FMMNode_t*>& nodes_out=n_list[5];
      setup_data.nodes_in .clear();
      setup_data.nodes_out.clear();
      for(size_t i=0;i<nodes_in .Dim();i++) if((nodes_in [i]->depth==level || level==-1) && nodes_in [i]->trg_coord.Dim() && nodes_in [i]->IsLeaf() && !nodes_in [i]->IsGhost()) setup_data.nodes_in .push_back(nodes_in [i]);
      for(size_t i=0;i<nodes_out.Dim();i++) if((nodes_out[i]->depth==level || level==-1) && nodes_out[i]->trg_coord.Dim() && nodes_out[i]->IsLeaf() && !nodes_out[i]->IsGhost()) setup_data.nodes_out.push_back(nodes_out[i]);
    }
    ptSetupData data;
    data. level=setup_data. level;
    data.kernel=setup_data.kernel;
    std::vector<FMMNode_t*>& nodes_in =setup_data.nodes_in ;
    std::vector<FMMNode_t*>& nodes_out=setup_data.nodes_out;
    {
      std::vector<FMMNode_t*>& nodes=nodes_in;
      PackedData& coord=data.src_coord;
      PackedData& value=data.src_value;
      coord.ptr=setup_data. coord_data;
      value.ptr=setup_data. input_data;
      coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
      value.len=value.ptr->Dim(0)*value.ptr->Dim(1);
      coord.cnt.ReInit(nodes.size());
      coord.dsp.ReInit(nodes.size());
      value.cnt.ReInit(nodes.size());
      value.dsp.ReInit(nodes.size());
#pragma omp parallel for
      for(size_t i=0;i<nodes.size();i++){
        nodes[i]->node_id=i;
        Vector<Real_t>& coord_vec=tree->dnwd_equiv_surf[nodes[i]->depth];
        Vector<Real_t>& value_vec=(nodes[i]->FMMData())->dnward_equiv;
        if(coord_vec.Dim()){
          coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
          assert(coord.dsp[i]<coord.len);
          coord.cnt[i]=coord_vec.Dim();
        }else{
          coord.dsp[i]=0;
          coord.cnt[i]=0;
        }
        if(value_vec.Dim()){
          value.dsp[i]=&value_vec[0]-value.ptr[0][0];
          assert(value.dsp[i]<value.len);
          value.cnt[i]=value_vec.Dim();
        }else{
          value.dsp[i]=0;
          value.cnt[i]=0;
        }
      }
    }
    {
      std::vector<FMMNode_t*>& nodes=nodes_in;
      PackedData& coord=data.srf_coord;
      PackedData& value=data.srf_value;
      coord.ptr=setup_data. coord_data;
      value.ptr=setup_data. input_data;
      coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
      value.len=value.ptr->Dim(0)*value.ptr->Dim(1);
      coord.cnt.ReInit(nodes.size());
      coord.dsp.ReInit(nodes.size());
      value.cnt.ReInit(nodes.size());
      value.dsp.ReInit(nodes.size());
#pragma omp parallel for
      for(size_t i=0;i<nodes.size();i++){
        coord.dsp[i]=0;
        coord.cnt[i]=0;
        value.dsp[i]=0;
        value.cnt[i]=0;
      }
    }
    {
      std::vector<FMMNode_t*>& nodes=nodes_out;
      PackedData& coord=data.trg_coord;
      PackedData& value=data.trg_value;
      coord.ptr=setup_data. coord_data;
      value.ptr=setup_data.output_data;
      coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
      value.len=value.ptr->Dim(0)*value.ptr->Dim(1);
      coord.cnt.ReInit(nodes.size());
      coord.dsp.ReInit(nodes.size());
      value.cnt.ReInit(nodes.size());
      value.dsp.ReInit(nodes.size());
#pragma omp parallel for
      for(size_t i=0;i<nodes.size();i++){
        Vector<Real_t>& coord_vec=nodes[i]->trg_coord;
        Vector<Real_t>& value_vec=nodes[i]->trg_value;
        if(coord_vec.Dim()){
          coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
          assert(coord.dsp[i]<coord.len);
          coord.cnt[i]=coord_vec.Dim();
        }else{
          coord.dsp[i]=0;
          coord.cnt[i]=0;
        }
        if(value_vec.Dim()){
          value.dsp[i]=&value_vec[0]-value.ptr[0][0];
          assert(value.dsp[i]<value.len);
          value.cnt[i]=value_vec.Dim();
        }else{
          value.dsp[i]=0;
          value.cnt[i]=0;
        }
      }
    }
    {
      int omp_p=omp_get_max_threads();
      std::vector<std::vector<size_t> > in_node_(omp_p);
      std::vector<std::vector<size_t> > scal_idx_(omp_p);
      std::vector<std::vector<Real_t> > coord_shift_(omp_p);
      std::vector<std::vector<size_t> > interac_cnt_(omp_p);
      if(this->ScaleInvar()){
        const Kernel<Real_t>* ker=kernel->k_l2l;
        for(size_t l=0;l<MAX_DEPTH;l++){
          Vector<Real_t>& scal=data.interac_data.scal[l*4+0];
          Vector<Real_t>& scal_exp=ker->trg_scal;
          scal.ReInit(scal_exp.Dim());
          for(size_t i=0;i<scal.Dim();i++){
            scal[i]=pvfmm::pow<Real_t>(2.0,-scal_exp[i]*l);
          }
        }
        for(size_t l=0;l<MAX_DEPTH;l++){
          Vector<Real_t>& scal=data.interac_data.scal[l*4+1];
          Vector<Real_t>& scal_exp=ker->src_scal;
          scal.ReInit(scal_exp.Dim());
          for(size_t i=0;i<scal.Dim();i++){
            scal[i]=pvfmm::pow<Real_t>(2.0,-scal_exp[i]*l);
          }
        }
      }
#pragma omp parallel for
      for(size_t tid=0;tid<omp_p;tid++){
        std::vector<size_t>& in_node    =in_node_[tid]    ;
        std::vector<size_t>& scal_idx   =scal_idx_[tid]   ;
        std::vector<Real_t>& coord_shift=coord_shift_[tid];
        std::vector<size_t>& interac_cnt=interac_cnt_[tid];
        size_t a=(nodes_out.size()*(tid+0))/omp_p;
        size_t b=(nodes_out.size()*(tid+1))/omp_p;
        for(size_t i=a;i<b;i++){
          FMMNode_t* tnode=nodes_out[i];
          Real_t s=pvfmm::pow<Real_t>(0.5,tnode->depth);
          size_t interac_cnt_=0;
          {
            Mat_Type type=D2T_Type;
            Vector<FMMNode_t*>& intlst=tnode->interac_list[type];
            for(size_t j=0;j<intlst.Dim();j++) if(intlst[j]){
              FMMNode_t* snode=intlst[j];
              size_t snode_id=snode->node_id;
              if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
              in_node.push_back(snode_id);
              scal_idx.push_back(snode->depth);
              {
                const int* rel_coord=interac_list.RelativeCoord(type,j);
                const Real_t* scoord=snode->Coord();
                const Real_t* tcoord=tnode->Coord();
                Real_t shift[COORD_DIM];
                shift[0]=rel_coord[0]*0.5*s-(0+0.5*s)+(tcoord[0]+0.5*s);
                shift[1]=rel_coord[1]*0.5*s-(0+0.5*s)+(tcoord[1]+0.5*s);
                shift[2]=rel_coord[2]*0.5*s-(0+0.5*s)+(tcoord[2]+0.5*s);
                coord_shift.push_back(shift[0]);
                coord_shift.push_back(shift[1]);
                coord_shift.push_back(shift[2]);
              }
              interac_cnt_++;
            }
          }
          interac_cnt.push_back(interac_cnt_);
        }
      }
      {
        InteracData& interac_data=data.interac_data;
	CopyVec(in_node_,interac_data.in_node);
	CopyVec(scal_idx_,interac_data.scal_idx);
	CopyVec(coord_shift_,interac_data.coord_shift);
	CopyVec(interac_cnt_,interac_data.interac_cnt);
        {
          pvfmm::Vector<size_t>& cnt=interac_data.interac_cnt;
          pvfmm::Vector<size_t>& dsp=interac_data.interac_dsp;
          dsp.ReInit(cnt.Dim()); if(dsp.Dim()) dsp[0]=0;
          omp_par::scan(&cnt[0],&dsp[0],dsp.Dim());
        }
      }
      {
        InteracData& interac_data=data.interac_data;
        pvfmm::Vector<size_t>& cnt=interac_data.interac_cnt;
        pvfmm::Vector<size_t>& dsp=interac_data.interac_dsp;
        if(cnt.Dim() && cnt[cnt.Dim()-1]+dsp[dsp.Dim()-1]){
          data.interac_data.M[0]=this->mat->Mat(level, DC2DE0_Type, 0);
          data.interac_data.M[1]=this->mat->Mat(level, DC2DE1_Type, 0);
        }else{
          data.interac_data.M[0].ReInit(0,0);
          data.interac_data.M[1].ReInit(0,0);
        }
      }
    }
    PtSetup(setup_data, &data);
  }

  void Down2Target(SetupData<Real_t,FMMNode_t>&  setup_data){
    if(!this->MultipoleOrder()) return;
    this->EvalListPts(setup_data);
  }
    
  void PostProcessing(FMMTree_t* tree, std::vector<FMMNode_t*>& nodes, BoundaryType bndry=FreeSpace){
    if(kernel->k_m2l->vol_poten && bndry==Periodic){
      const Kernel<Real_t>& k_m2t=*kernel->k_m2t;
      int ker_dim[2]={k_m2t.ker_dim[0],k_m2t.ker_dim[1]};
      Vector<Real_t>& up_equiv=(tree->RootNode()->FMMData())->upward_equiv;
      Matrix<Real_t> avg_density(1,ker_dim[0]); avg_density.SetZero();
      for(size_t i0=0;i0<up_equiv.Dim();i0+=ker_dim[0]){
        for(size_t i1=0;i1<ker_dim[0];i1++){
          avg_density[0][i1]+=up_equiv[i0+i1];
        }
      }
      int omp_p=omp_get_max_threads();
      std::vector<Matrix<Real_t> > M_tmp(omp_p);
#pragma omp parallel for
      for(size_t i=0;i<nodes.size();i++)
      if(nodes[i]->IsLeaf() && !nodes[i]->IsGhost()){
        Vector<Real_t>& trg_coord=nodes[i]->trg_coord;
        Vector<Real_t>& trg_value=nodes[i]->trg_value;
        size_t n_trg=trg_coord.Dim()/COORD_DIM;
        Matrix<Real_t>& M_vol=M_tmp[omp_get_thread_num()];
        M_vol.ReInit(ker_dim[0],n_trg*ker_dim[1]); M_vol.SetZero();
        k_m2t.vol_poten(&trg_coord[0],n_trg,&M_vol[0][0]);
        Matrix<Real_t> M_trg(1,n_trg*ker_dim[1],&trg_value[0],false);
        M_trg-=avg_density*M_vol;
      }
    }
  }      
  
  void DownwardPass() {
    Profile::Tic("Setup",true,3);
    std::vector<FMMNode_t*> leaf_nodes;
    int max_depth=0;
    int max_depth_loc=0;
    std::vector<FMMNode_t*>& nodes=this->GetNodeList();
    for(size_t i=0;i<nodes.size();i++){
      FMMNode_t* n=nodes[i];
      if(!n->IsGhost() && n->IsLeaf()) leaf_nodes.push_back(n);
      if(n->depth>max_depth_loc) max_depth_loc=n->depth;
    }
    max_depth = max_depth_loc;
    Profile::Toc();
    if(bndry==Periodic) {
      Profile::Tic("BoundaryCondition",false,5);
      this->PeriodicBC(dynamic_cast<FMMNode_t*>(this->RootNode()));
      Profile::Toc();
    }
    for(size_t i=0; i<=(this->ScaleInvar()?0:max_depth); i++) {
      if(!this->ScaleInvar()) {
        std::stringstream level_str;
        level_str<<"Level-"<<std::setfill('0')<<std::setw(2)<<i<<"\0";
        Profile::Tic(level_str.str().c_str(),false,5);
        Profile::Tic("Precomp",false,5);
	{Profile::Tic("Precomp-U",false,10);
        this->SetupPrecomp(setup_data[i+MAX_DEPTH*0]);
        Profile::Toc();}
        {Profile::Tic("Precomp-W",false,10);
        this->SetupPrecomp(setup_data[i+MAX_DEPTH*1]);
        Profile::Toc();}
        {Profile::Tic("Precomp-X",false,10);
        this->SetupPrecomp(setup_data[i+MAX_DEPTH*2]);
        Profile::Toc();}
        if(0){
          Profile::Tic("Precomp-V",false,10);
          this->SetupPrecomp(setup_data[i+MAX_DEPTH*3]);
          Profile::Toc();
        }
	Profile::Toc();
      }
      {Profile::Tic("X-List",false,5);
      this->X_List(setup_data[i+MAX_DEPTH*2]);
      Profile::Toc();}
      {Profile::Tic("W-List",false,5);
      this->W_List(setup_data[i+MAX_DEPTH*1]);
      Profile::Toc();}
      {Profile::Tic("U-List",false,5);
      this->U_List(setup_data[i+MAX_DEPTH*0]);
      Profile::Toc();}
      {Profile::Tic("V-List",false,5);
      this->V_List(setup_data[i+MAX_DEPTH*3]);
      Profile::Toc();}
      if(!this->ScaleInvar()){
        Profile::Toc();
      }
    }
    Profile::Tic("D2D",false,5);
    for(size_t i=0; i<=max_depth; i++) {
      if(!this->ScaleInvar()) this->SetupPrecomp(setup_data[i+MAX_DEPTH*4]);
      this->Down2Down(setup_data[i+MAX_DEPTH*4]);
    }
    Profile::Toc();
    Profile::Tic("D2T",false,5);
    for(int i=0; i<=(this->ScaleInvar()?0:max_depth); i++) {
      if(!this->ScaleInvar()) this->SetupPrecomp(setup_data[i+MAX_DEPTH*5]);
      this->Down2Target(setup_data[i+MAX_DEPTH*5]);
    }
    Profile::Toc();
    Profile::Tic("PostProc",false,5);
    this->PostProcessing(this, leaf_nodes, bndry);
    Profile::Toc();
  }

};

}//end namespace

#endif //_PVFMM_FMM_TREE_HPP_

