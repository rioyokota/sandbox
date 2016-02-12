namespace pvfmm{

template <class Real_t>
void FMM_Data<Real_t>::Clear(){
  upward_equiv.Resize(0);
}

template <class FMMNode_t>
FMM_Pts<FMMNode_t>::~FMM_Pts() {
  if(mat!=NULL){
    delete mat;
    mat=NULL;
  }
  if(vprecomp_fft_flag) FFTW_t<Real_t>::fft_destroy_plan(vprecomp_fftplan);
  {
    if(vlist_fft_flag ) FFTW_t<Real_t>::fft_destroy_plan(vlist_fftplan );
    if(vlist_ifft_flag) FFTW_t<Real_t>::fft_destroy_plan(vlist_ifftplan);
    vlist_fft_flag =false;
    vlist_ifft_flag=false;
  }
}

template <class FMMNode_t>
void FMM_Pts<FMMNode_t>::Down2DownSetup(SetupData<Real_t>& setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level){
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
  std::vector<void*>& nodes_in =setup_data.nodes_in ;
  std::vector<void*>& nodes_out=setup_data.nodes_out;
  std::vector<Vector<Real_t>*>&  input_vector=setup_data. input_vector;  input_vector.clear();
  std::vector<Vector<Real_t>*>& output_vector=setup_data.output_vector; output_vector.clear();
  for(size_t i=0;i<nodes_in .size();i++)  input_vector.push_back(&((FMMData*)((FMMNode_t*)nodes_in [i])->FMMData())->dnward_equiv);
  for(size_t i=0;i<nodes_out.size();i++) output_vector.push_back(&((FMMData*)((FMMNode_t*)nodes_out[i])->FMMData())->dnward_equiv);
  SetupInterac(setup_data);
}

template <class FMMNode_t>
void FMM_Pts<FMMNode_t>::Down2Down     (SetupData<Real_t>& setup_data){
  if(!this->MultipoleOrder()) return;
  EvalList(setup_data);
}

template <class FMMNode_t>
void FMM_Pts<FMMNode_t>::X_ListSetup(SetupData<Real_t>&  setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level){
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
  struct PackedData{
    size_t len;
    Matrix<Real_t>* ptr;
    Vector<size_t> cnt;
    Vector<size_t> dsp;
  };
  struct InteracData{
    Vector<size_t> in_node;
    Vector<size_t> scal_idx;
    Vector<Real_t> coord_shift;
    Vector<size_t> interac_cnt;
    Vector<size_t> interac_dsp;
    Vector<size_t> interac_cst;
    Vector<Real_t> scal[4*MAX_DEPTH];
    Matrix<Real_t> M[4];
  };
  struct ptSetupData{
    int level;
    const Kernel<Real_t>* kernel;
    PackedData src_coord;
    PackedData src_value;
    PackedData srf_coord;
    PackedData srf_value;
    PackedData trg_coord;
    PackedData trg_value;
    InteracData interac_data;
  };
  ptSetupData data;
  data. level=setup_data. level;
  data.kernel=setup_data.kernel;
  std::vector<void*>& nodes_in =setup_data.nodes_in ;
  std::vector<void*>& nodes_out=setup_data.nodes_out;
  {
    std::vector<void*>& nodes=nodes_in;
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
      Vector<Real_t>& coord_vec=((FMMNode_t*)nodes[i])->src_coord;
      Vector<Real_t>& value_vec=((FMMNode_t*)nodes[i])->src_value;
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
    std::vector<void*>& nodes=nodes_in;
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
      Vector<Real_t>& coord_vec=((FMMNode_t*)nodes[i])->surf_coord;
      Vector<Real_t>& value_vec=((FMMNode_t*)nodes[i])->surf_value;
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
    std::vector<void*>& nodes=nodes_out;
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
      Vector<Real_t>& coord_vec=tree->dnwd_check_surf[((FMMNode_t*)nodes[i])->depth];
      Vector<Real_t>& value_vec=((FMMData*)((FMMNode_t*)nodes[i])->FMMData())->dnward_equiv;
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
        FMMNode_t* tnode=(FMMNode_t*)nodes_out[i];
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
      {
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=in_node_;
        pvfmm::Vector<ElemType>& vec=interac_data.in_node;
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
      {
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=scal_idx_;
        pvfmm::Vector<ElemType>& vec=interac_data.scal_idx;
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
      {
        typedef Real_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=coord_shift_;
        pvfmm::Vector<ElemType>& vec=interac_data.coord_shift;
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
      {
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=interac_cnt_;
        pvfmm::Vector<ElemType>& vec=interac_data.interac_cnt;
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

template <class FMMNode_t>
void FMM_Pts<FMMNode_t>::X_List     (SetupData<Real_t>&  setup_data){
  if(!this->MultipoleOrder()) return;
  this->EvalListPts(setup_data);
}


template <class FMMNode_t>
void FMM_Pts<FMMNode_t>::W_ListSetup(SetupData<Real_t>&  setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level){
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
  struct PackedData{
    size_t len;
    Matrix<Real_t>* ptr;
    Vector<size_t> cnt;
    Vector<size_t> dsp;
  };
  struct InteracData{
    Vector<size_t> in_node;
    Vector<size_t> scal_idx;
    Vector<Real_t> coord_shift;
    Vector<size_t> interac_cnt;
    Vector<size_t> interac_dsp;
    Vector<size_t> interac_cst;
    Vector<Real_t> scal[4*MAX_DEPTH];
    Matrix<Real_t> M[4];
  };
  struct ptSetupData{
    int level;
    const Kernel<Real_t>* kernel;
    PackedData src_coord;
    PackedData src_value;
    PackedData srf_coord;
    PackedData srf_value;
    PackedData trg_coord;
    PackedData trg_value;
    InteracData interac_data;
  };
  ptSetupData data;
  data. level=setup_data. level;
  data.kernel=setup_data.kernel;
  std::vector<void*>& nodes_in =setup_data.nodes_in ;
  std::vector<void*>& nodes_out=setup_data.nodes_out;
  {
    std::vector<void*>& nodes=nodes_in;
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
      Vector<Real_t>& coord_vec=tree->upwd_equiv_surf[((FMMNode_t*)nodes[i])->depth];
      Vector<Real_t>& value_vec=((FMMData*)((FMMNode_t*)nodes[i])->FMMData())->upward_equiv;
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
    std::vector<void*>& nodes=nodes_in;
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
    std::vector<void*>& nodes=nodes_out;
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
      Vector<Real_t>& coord_vec=((FMMNode_t*)nodes[i])->trg_coord;
      Vector<Real_t>& value_vec=((FMMNode_t*)nodes[i])->trg_value;
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
        FMMNode_t* tnode=(FMMNode_t*)nodes_out[i];
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
      {
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=in_node_;
        pvfmm::Vector<ElemType>& vec=interac_data.in_node;
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
      {
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=scal_idx_;
        pvfmm::Vector<ElemType>& vec=interac_data.scal_idx;
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
      {
        typedef Real_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=coord_shift_;
        pvfmm::Vector<ElemType>& vec=interac_data.coord_shift;
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
      {
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=interac_cnt_;
        pvfmm::Vector<ElemType>& vec=interac_data.interac_cnt;
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

template <class FMMNode_t>
void FMM_Pts<FMMNode_t>::W_List     (SetupData<Real_t>&  setup_data){
  if(!this->MultipoleOrder()) return;
  this->EvalListPts(setup_data);
}


template <class FMMNode_t>
void FMM_Pts<FMMNode_t>::U_ListSetup(SetupData<Real_t>& setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level){
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
  struct PackedData{
    size_t len;
    Matrix<Real_t>* ptr;
    Vector<size_t> cnt;
    Vector<size_t> dsp;
  };
  struct InteracData{
    Vector<size_t> in_node;
    Vector<size_t> scal_idx;
    Vector<Real_t> coord_shift;
    Vector<size_t> interac_cnt;
    Vector<size_t> interac_dsp;
    Vector<size_t> interac_cst;
    Vector<Real_t> scal[4*MAX_DEPTH];
    Matrix<Real_t> M[4];
  };
  struct ptSetupData{
    int level;
    const Kernel<Real_t>* kernel;
    PackedData src_coord;
    PackedData src_value;
    PackedData srf_coord;
    PackedData srf_value;
    PackedData trg_coord;
    PackedData trg_value;
    InteracData interac_data;
  };
  ptSetupData data;
  data. level=setup_data. level;
  data.kernel=setup_data.kernel;
  std::vector<void*>& nodes_in =setup_data.nodes_in ;
  std::vector<void*>& nodes_out=setup_data.nodes_out;
  {
    std::vector<void*>& nodes=nodes_in;
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
      Vector<Real_t>& coord_vec=((FMMNode_t*)nodes[i])->src_coord;
      Vector<Real_t>& value_vec=((FMMNode_t*)nodes[i])->src_value;
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
    std::vector<void*>& nodes=nodes_in;
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
      Vector<Real_t>& coord_vec=((FMMNode_t*)nodes[i])->surf_coord;
      Vector<Real_t>& value_vec=((FMMNode_t*)nodes[i])->surf_value;
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
    std::vector<void*>& nodes=nodes_out;
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
      Vector<Real_t>& coord_vec=((FMMNode_t*)nodes[i])->trg_coord;
      Vector<Real_t>& value_vec=((FMMNode_t*)nodes[i])->trg_value;
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
        FMMNode_t* tnode=(FMMNode_t*)nodes_out[i];
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
      {
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=in_node_;
        pvfmm::Vector<ElemType>& vec=interac_data.in_node;
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
      {
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=scal_idx_;
        pvfmm::Vector<ElemType>& vec=interac_data.scal_idx;
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
      {
        typedef Real_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=coord_shift_;
        pvfmm::Vector<ElemType>& vec=interac_data.coord_shift;
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
      {
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=interac_cnt_;
        pvfmm::Vector<ElemType>& vec=interac_data.interac_cnt;
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

template <class FMMNode_t>
void FMM_Pts<FMMNode_t>::U_List     (SetupData<Real_t>&  setup_data){
  this->EvalListPts(setup_data);
}

template <class FMMNode_t>
void FMM_Pts<FMMNode_t>::Down2TargetSetup(SetupData<Real_t>&  setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level){
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
  struct PackedData{
    size_t len;
    Matrix<Real_t>* ptr;
    Vector<size_t> cnt;
    Vector<size_t> dsp;
  };
  struct InteracData{
    Vector<size_t> in_node;
    Vector<size_t> scal_idx;
    Vector<Real_t> coord_shift;
    Vector<size_t> interac_cnt;
    Vector<size_t> interac_dsp;
    Vector<size_t> interac_cst;
    Vector<Real_t> scal[4*MAX_DEPTH];
    Matrix<Real_t> M[4];
  };
  struct ptSetupData{
    int level;
    const Kernel<Real_t>* kernel;
    PackedData src_coord;
    PackedData src_value;
    PackedData srf_coord;
    PackedData srf_value;
    PackedData trg_coord;
    PackedData trg_value;
    InteracData interac_data;
  };
  ptSetupData data;
  data. level=setup_data. level;
  data.kernel=setup_data.kernel;
  std::vector<void*>& nodes_in =setup_data.nodes_in ;
  std::vector<void*>& nodes_out=setup_data.nodes_out;
  {
    std::vector<void*>& nodes=nodes_in;
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
      Vector<Real_t>& coord_vec=tree->dnwd_equiv_surf[((FMMNode_t*)nodes[i])->depth];
      Vector<Real_t>& value_vec=((FMMData*)((FMMNode_t*)nodes[i])->FMMData())->dnward_equiv;
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
    std::vector<void*>& nodes=nodes_in;
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
    std::vector<void*>& nodes=nodes_out;
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
      Vector<Real_t>& coord_vec=((FMMNode_t*)nodes[i])->trg_coord;
      Vector<Real_t>& value_vec=((FMMNode_t*)nodes[i])->trg_value;
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
        FMMNode_t* tnode=(FMMNode_t*)nodes_out[i];
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
      {
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=in_node_;
        pvfmm::Vector<ElemType>& vec=interac_data.in_node;
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
      {
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=scal_idx_;
        pvfmm::Vector<ElemType>& vec=interac_data.scal_idx;
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
      {
        typedef Real_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=coord_shift_;
        pvfmm::Vector<ElemType>& vec=interac_data.coord_shift;
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
      {
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=interac_cnt_;
        pvfmm::Vector<ElemType>& vec=interac_data.interac_cnt;
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

template <class FMMNode_t>
void FMM_Pts<FMMNode_t>::Down2Target(SetupData<Real_t>&  setup_data){
  if(!this->MultipoleOrder()) return;
  this->EvalListPts(setup_data);
}


template <class FMMNode_t>
void FMM_Pts<FMMNode_t>::PostProcessing(FMMTree_t* tree, std::vector<FMMNode_t*>& nodes, BoundaryType bndry){
  if(kernel->k_m2l->vol_poten && bndry==Periodic){
    const Kernel<Real_t>& k_m2t=*kernel->k_m2t;
    int ker_dim[2]={k_m2t.ker_dim[0],k_m2t.ker_dim[1]};
    Vector<Real_t>& up_equiv=((FMMData*)tree->RootNode()->FMMData())->upward_equiv;
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

}//end namespace
