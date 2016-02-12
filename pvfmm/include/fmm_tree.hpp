#include <mpi_tree.hpp>

#ifndef _PVFMM_FMM_TREE_HPP_
#define _PVFMM_FMM_TREE_HPP_

namespace pvfmm{

template <class FMM_Mat_t>
class FMM_Tree: public MPI_Tree<typename FMM_Mat_t::FMMNode_t>{

 public:

  typedef typename FMM_Mat_t::FMMNode_t TreeNode;

  std::vector<Matrix<Real_t> > node_data_buff;
  pvfmm::Matrix<TreeNode*> node_interac_lst;
  InteracList<TreeNode> interac_list;
  FMM_Mat_t* fmm_mat;
  BoundaryType bndry;
  std::vector<Matrix<char> > precomp_lst;
  std::vector<SetupData<Real_t> > setup_data;
  std::vector<Vector<Real_t> > upwd_check_surf;
  std::vector<Vector<Real_t> > upwd_equiv_surf;
  std::vector<Vector<Real_t> > dnwd_check_surf;
  std::vector<Vector<Real_t> > dnwd_equiv_surf;

  FMM_Tree(): MPI_Tree<TreeNode>(), fmm_mat(NULL), bndry(FreeSpace) { };

  virtual ~FMM_Tree(){}

  virtual void Initialize(typename TreeNode::NodeData* init_data) {
    Profile::Tic("InitTree",true);{
      MPI_Tree<TreeNode>::Initialize(init_data);
      Profile::Tic("InitFMMData",true,5);{
	std::vector<TreeNode*>& nodes=this->GetNodeList();
#pragma omp parallel for
	for(size_t i=0;i<nodes.size();i++){
	  if(nodes[i]->FMMData()==NULL) nodes[i]->FMMData()=mem::aligned_new<typename FMM_Mat_t::FMMData>();
	}
      }Profile::Toc();   
    }Profile::Toc();
  }

  void SetColleagues(BoundaryType bndry=FreeSpace, TreeNode* node=NULL) {
    int n1=(int)pvfmm::pow<unsigned int>(3,this->Dim());
    int n2=(int)pvfmm::pow<unsigned int>(2,this->Dim());
    if(node==NULL){
      TreeNode* curr_node=this->PreorderFirst();
      if(curr_node!=NULL){
        if(bndry==Periodic){
          for(int i=0;i<n1;i++)
            curr_node->SetColleague(curr_node,i);
        }else{
          curr_node->SetColleague(curr_node,(n1-1)/2);
        }
        curr_node=this->PreorderNxt(curr_node);
      }
      Vector<std::vector<TreeNode*> > nodes(MAX_DEPTH);
      while(curr_node!=NULL){
        nodes[curr_node->depth].push_back(curr_node);
        curr_node=this->PreorderNxt(curr_node);
      }
      for(size_t i=0;i<MAX_DEPTH;i++){
        size_t j0=nodes[i].size();
        TreeNode** nodes_=&nodes[i][0];
#pragma omp parallel for
        for(size_t j=0;j<j0;j++){
          SetColleagues(bndry, nodes_[j]);
        }
      }
    }else{
      TreeNode* parent_node;
      TreeNode* tmp_node1;
      TreeNode* tmp_node2;
      for(int i=0;i<n1;i++)node->SetColleague(NULL,i);
      parent_node=(TreeNode*)node->Parent();
      if(parent_node==NULL) return;
      int l=node->Path2Node();
      for(int i=0;i<n1;i++){
        tmp_node1=(TreeNode*)parent_node->Colleague(i);
        if(tmp_node1!=NULL)
        if(!tmp_node1->IsLeaf()){
          for(int j=0;j<n2;j++){
            tmp_node2=(TreeNode*)tmp_node1->Child(j);
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

  void InitFMM_Tree(bool refine, BoundaryType bndry_=FreeSpace) {
    Profile::Tic("InitFMM_Tree",true);{
      interac_list.Initialize(this->Dim());
      bndry=bndry_;
    }Profile::Toc();
  }

  void SetupFMM(FMM_Mat_t* fmm_mat_) {
    Profile::Tic("SetupFMM",true);{
    typedef typename FMM_Mat_t::FMMTree_t MatTree_t;
    if(fmm_mat!=fmm_mat_) {
      setup_data.clear();
      precomp_lst.clear();
      fmm_mat=fmm_mat_;
    }
    Profile::Tic("SetColleagues",false,3);
    this->SetColleagues(bndry);
    Profile::Toc();
    Profile::Tic("CollectNodeData",false,3);
    TreeNode* n=dynamic_cast<TreeNode*>(this->PostorderFirst());
    std::vector<TreeNode*> all_nodes;
    while(n!=NULL){
      n->pt_cnt[0]=0;
      n->pt_cnt[1]=0;
      all_nodes.push_back(n);
      n=static_cast<TreeNode*>(this->PostorderNxt(n));
    }
    std::vector<Vector<TreeNode*> > node_lists; // TODO: Remove this parameter, not really needed
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
      std::vector<TreeNode*>& nodes=this->GetNodeList();
      for(size_t i=0;i<nodes.size();i++){
        TreeNode* n=nodes[i];
        if(n->depth>max_depth_loc) max_depth_loc=n->depth;
      }
      max_depth = max_depth_loc;
    }
    Profile::Tic("S2U",false,5);
    for(int i=0; i<=(fmm_mat->ScaleInvar()?0:max_depth); i++){
      if(!fmm_mat->ScaleInvar()) fmm_mat->SetupPrecomp(setup_data[i+MAX_DEPTH*6]);
      fmm_mat->Source2Up(setup_data[i+MAX_DEPTH*6]);
    }
    Profile::Toc();
    Profile::Tic("U2U",false,5);
    for(int i=max_depth-1; i>=0; i--){
      if(!fmm_mat->ScaleInvar()) fmm_mat->SetupPrecomp(setup_data[i+MAX_DEPTH*7]);
      fmm_mat->Up2Up(setup_data[i+MAX_DEPTH*7]);
    }
    Profile::Toc();
  }
  
  void BuildInteracLists() {
    std::vector<TreeNode*> n_list_src;
    std::vector<TreeNode*> n_list_trg;
    {
      std::vector<TreeNode*>& nodes=this->GetNodeList();
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
    std::vector<std::vector<TreeNode*>*> type_node_lst;
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
        std::vector<TreeNode*>& n_list=*type_node_lst[k];
        size_t a=(n_list.size()*(j  ))/omp_p;
        size_t b=(n_list.size()*(j+1))/omp_p;
        for(size_t i=a;i<b;i++){
          TreeNode* n=n_list[i];
          n->interac_list[type_lst[k]].ReInit(interac_cnt[k],&node_interac_lst[i][interac_dsp[k]],false);
          interac_list.BuildList(n,type_lst[k]);
        }
      }
    }
  }
  
  void DownwardPass() {
    Profile::Tic("Setup",true,3);
    std::vector<TreeNode*> leaf_nodes;
    int max_depth=0;
    {
      int max_depth_loc=0;
      std::vector<TreeNode*>& nodes=this->GetNodeList();
      for(size_t i=0;i<nodes.size();i++){
        TreeNode* n=nodes[i];
        if(!n->IsGhost() && n->IsLeaf()) leaf_nodes.push_back(n);
        if(n->depth>max_depth_loc) max_depth_loc=n->depth;
      }
      max_depth = max_depth_loc;
    }
    Profile::Toc();
    if(bndry==Periodic) {
      Profile::Tic("BoundaryCondition",false,5);
      fmm_mat->PeriodicBC(dynamic_cast<TreeNode*>(this->RootNode()));
      Profile::Toc();
    }
    for(size_t i=0; i<=(fmm_mat->ScaleInvar()?0:max_depth); i++) {
      if(!fmm_mat->ScaleInvar()) {
        std::stringstream level_str;
        level_str<<"Level-"<<std::setfill('0')<<std::setw(2)<<i<<"\0";
        Profile::Tic(level_str.str().c_str(),false,5);
        Profile::Tic("Precomp",false,5);
        {
          Profile::Tic("Precomp-U",false,10);
          fmm_mat->SetupPrecomp(setup_data[i+MAX_DEPTH*0]);
          Profile::Toc();
        }
        {
          Profile::Tic("Precomp-W",false,10);
          fmm_mat->SetupPrecomp(setup_data[i+MAX_DEPTH*1]);
          Profile::Toc();
        }
        {
          Profile::Tic("Precomp-X",false,10);
          fmm_mat->SetupPrecomp(setup_data[i+MAX_DEPTH*2]);
          Profile::Toc();
        }
        if(0){
          Profile::Tic("Precomp-V",false,10);
          fmm_mat->SetupPrecomp(setup_data[i+MAX_DEPTH*3]);
          Profile::Toc();
        }
        Profile::Toc();
      }
      {
        Profile::Tic("X-List",false,5);
        fmm_mat->X_List(setup_data[i+MAX_DEPTH*2]);
        Profile::Toc();
      }
      {
        Profile::Tic("W-List",false,5);
        fmm_mat->W_List(setup_data[i+MAX_DEPTH*1]);
        Profile::Toc();
      }
      {
        Profile::Tic("U-List",false,5);
        fmm_mat->U_List(setup_data[i+MAX_DEPTH*0]);
        Profile::Toc();
      }
      {
        Profile::Tic("V-List",false,5);
        fmm_mat->V_List(setup_data[i+MAX_DEPTH*3]);
        Profile::Toc();
      }
      if(!fmm_mat->ScaleInvar()){
        Profile::Toc();
      }
    }
    Profile::Tic("D2D",false,5);
    for(size_t i=0; i<=max_depth; i++) {
      if(!fmm_mat->ScaleInvar()) fmm_mat->SetupPrecomp(setup_data[i+MAX_DEPTH*4]);
      fmm_mat->Down2Down(setup_data[i+MAX_DEPTH*4]);
    }
    Profile::Toc();
    Profile::Tic("D2T",false,5);
    for(int i=0; i<=(fmm_mat->ScaleInvar()?0:max_depth); i++) {
      if(!fmm_mat->ScaleInvar()) fmm_mat->SetupPrecomp(setup_data[i+MAX_DEPTH*5]);
      fmm_mat->Down2Target(setup_data[i+MAX_DEPTH*5]);
    }
    Profile::Toc();
    Profile::Tic("PostProc",false,5);
    typedef typename FMM_Mat_t::FMMTree_t MatTree_t;
    fmm_mat->PostProcessing((MatTree_t*)this, leaf_nodes, bndry);
    Profile::Toc();
  }

};

}//end namespace

#endif //_PVFMM_FMM_TREE_HPP_

