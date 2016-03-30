#ifndef _PVFMM_INTERAC_LIST_HPP_
#define _PVFMM_INTERAC_LIST_HPP_

namespace pvfmm{

class InteracList{
public:

  std::vector<Matrix<int> > rel_coord;
  std::vector<std::vector<int> > hash_lut;
  std::vector<std::vector<size_t> > interac_class;
  std::vector<std::vector<std::vector<Perm_Type> > > perm_list;
  PrecompMat* mat;

  InteracList(){}

  void Initialize(PrecompMat* mat_=NULL){
    mat=mat_;
    interac_class.resize(Type_Count);
    perm_list.resize(Type_Count);
    rel_coord.resize(Type_Count);
    hash_lut.resize(Type_Count);

    InitList(0,0,1,UC2UE0_Type);
    InitList(0,0,1,UC2UE1_Type);
    InitList(0,0,1,DC2DE0_Type);
    InitList(0,0,1,DC2DE1_Type);

    InitList(0,0,1,S2U_Type);
    InitList(1,1,2,U2U_Type);
    InitList(1,1,2,D2D_Type);
    InitList(0,0,1,D2T_Type);

    InitList(3,3,2,U0_Type);
    InitList(1,0,1,U1_Type);
    InitList(3,3,2,U2_Type);

    InitList(3,2,1,V_Type);
    InitList(1,1,1,V1_Type);
    InitList(5,5,2,W_Type);
    InitList(5,5,2,X_Type);
    InitList(0,0,1,BC_Type);
  }

  size_t ListCount(Mat_Type t){
    return rel_coord[t].Dim(0);
  }

  int* RelativeCoord(Mat_Type t, size_t i){
    return rel_coord[t][i];
  }

  size_t InteracClass(Mat_Type t, size_t i){
    return interac_class[t][i];
  }

  std::vector<Perm_Type>& PermutList(Mat_Type t, size_t i){
    return perm_list[t][i];
  }

  void BuildList(FMM_Node* n, Mat_Type t){
    std::vector<FMM_Node*>& interac_list=n->interac_list[t];
    for (int i=0; i<interac_list.size(); i++) {
      interac_list[i] = 0;
    }

    static const int n_collg=27;
    static const int n_child=8;
    int rel_coord[3];

    switch (t){

    case S2U_Type:
      {
	if(!n->IsGhost() && n->IsLeaf()) interac_list[0]=n;
	break;
      }
    case U2U_Type:
      {
	if(n->IsGhost() || n->IsLeaf()) return;
	for(int j=0;j<n_child;j++){
	  rel_coord[0]=-1+(j & 1?2:0);
	  rel_coord[1]=-1+(j & 2?2:0);
	  rel_coord[2]=-1+(j & 4?2:0);
	  int c_hash = coord_hash(rel_coord);
	  int idx=hash_lut[t][c_hash];
	  FMM_Node* chld=(FMM_Node*)n->Child(j);
	  if(idx>=0 && !chld->IsGhost()) interac_list[idx]=chld;
	}
	break;
      }
    case D2D_Type:
      {
	if(n->IsGhost() || n->Parent()==NULL) return;
	FMM_Node* p=(FMM_Node*)n->Parent();
	int p2n=n->Path2Node();
	{
	  rel_coord[0]=-1+(p2n & 1?2:0);
	  rel_coord[1]=-1+(p2n & 2?2:0);
	  rel_coord[2]=-1+(p2n & 4?2:0);
	  int c_hash = coord_hash(rel_coord);
	  int idx=hash_lut[t][c_hash];
	  if(idx>=0) interac_list[idx]=p;
	}
	break;
      }
    case D2T_Type:
      {
	if(!n->IsGhost() && n->IsLeaf()) interac_list[0]=n;
	break;
      }
    case U0_Type:
      {
	if(n->IsGhost() || n->Parent()==NULL || !n->IsLeaf()) return;
	FMM_Node* p=(FMM_Node*)n->Parent();
	int p2n=n->Path2Node();
	for(int i=0;i<n_collg;i++){
	  FMM_Node* pc=(FMM_Node*)p->Colleague(i);
	  if(pc!=NULL && pc->IsLeaf()){
	    rel_coord[0]=( i %3)*4-4-(p2n & 1?2:0)+1;
	    rel_coord[1]=((i/3)%3)*4-4-(p2n & 2?2:0)+1;
	    rel_coord[2]=((i/9)%3)*4-4-(p2n & 4?2:0)+1;
	    int c_hash = coord_hash(rel_coord);
	    int idx=hash_lut[t][c_hash];
	    if(idx>=0) interac_list[idx]=pc;
	  }
	}
	break;
      }
    case U1_Type:
      {
	if(n->IsGhost() || !n->IsLeaf()) return;
	for(int i=0;i<n_collg;i++){
	  FMM_Node* col=(FMM_Node*)n->Colleague(i);
	  if(col!=NULL && col->IsLeaf()){
            rel_coord[0]=( i %3)-1;
            rel_coord[1]=((i/3)%3)-1;
            rel_coord[2]=((i/9)%3)-1;
            int c_hash = coord_hash(rel_coord);
            int idx=hash_lut[t][c_hash];
            if(idx>=0) interac_list[idx]=col;
	  }
	}
	break;
      }
    case U2_Type:
      {
	if(n->IsGhost() || !n->IsLeaf()) return;
	for(int i=0;i<n_collg;i++){
	  FMM_Node* col=(FMM_Node*)n->Colleague(i);
	  if(col!=NULL && !col->IsLeaf()){
	    for(int j=0;j<n_child;j++){
	      rel_coord[0]=( i %3)*4-4+(j & 1?2:0)-1;
	      rel_coord[1]=((i/3)%3)*4-4+(j & 2?2:0)-1;
	      rel_coord[2]=((i/9)%3)*4-4+(j & 4?2:0)-1;
	      int c_hash = coord_hash(rel_coord);
	      int idx=hash_lut[t][c_hash];
	      if(idx>=0){
		assert(col->Child(j)->IsLeaf()); //2:1 balanced
		interac_list[idx]=(FMM_Node*)col->Child(j);
	      }
	    }
	  }
	}
	break;
      }
    case V_Type:
      {
	if(n->IsGhost() || n->Parent()==NULL) return;
	FMM_Node* p=(FMM_Node*)n->Parent();
	int p2n=n->Path2Node();
	for(int i=0;i<n_collg;i++){
	  FMM_Node* pc=(FMM_Node*)p->Colleague(i);
	  if(pc!=NULL?!pc->IsLeaf():0){
	    for(int j=0;j<n_child;j++){
	      rel_coord[0]=( i   %3)*2-2+(j & 1?1:0)-(p2n & 1?1:0);
	      rel_coord[1]=((i/3)%3)*2-2+(j & 2?1:0)-(p2n & 2?1:0);
	      rel_coord[2]=((i/9)%3)*2-2+(j & 4?1:0)-(p2n & 4?1:0);
	      int c_hash = coord_hash(rel_coord);
	      int idx=hash_lut[t][c_hash];
	      if(idx>=0) interac_list[idx]=(FMM_Node*)pc->Child(j);
	    }
	  }
	}
	break;
      }
    case V1_Type:
      {
	if(n->IsGhost() || n->IsLeaf()) return;
	for(int i=0;i<n_collg;i++){
	  FMM_Node* col=(FMM_Node*)n->Colleague(i);
	  if(col!=NULL && !col->IsLeaf()){
            rel_coord[0]=( i %3)-1;
            rel_coord[1]=((i/3)%3)-1;
            rel_coord[2]=((i/9)%3)-1;
            int c_hash = coord_hash(rel_coord);
            int idx=hash_lut[t][c_hash];
            if(idx>=0) interac_list[idx]=col;
	  }
	}
	break;
      }
    case W_Type:
      {
	if(n->IsGhost() || !n->IsLeaf()) return;
	for(int i=0;i<n_collg;i++){
	  FMM_Node* col=(FMM_Node*)n->Colleague(i);
	  if(col!=NULL && !col->IsLeaf()){
	    for(int j=0;j<n_child;j++){
	      rel_coord[0]=( i %3)*4-4+(j & 1?2:0)-1;
	      rel_coord[1]=((i/3)%3)*4-4+(j & 2?2:0)-1;
	      rel_coord[2]=((i/9)%3)*4-4+(j & 4?2:0)-1;
	      int c_hash = coord_hash(rel_coord);
	      int idx=hash_lut[t][c_hash];
	      if(idx>=0) interac_list[idx]=(FMM_Node*)col->Child(j);
	    }
	  }
	}
	break;
      }
    case X_Type:
      {
	if(n->IsGhost() || n->Parent()==NULL) return;
	FMM_Node* p=(FMM_Node*)n->Parent();
	int p2n=n->Path2Node();
	for(int i=0;i<n_collg;i++){
	  FMM_Node* pc=(FMM_Node*)p->Colleague(i);
	  if(pc!=NULL && pc->IsLeaf()){
	    rel_coord[0]=( i %3)*4-4-(p2n & 1?2:0)+1;
	    rel_coord[1]=((i/3)%3)*4-4-(p2n & 2?2:0)+1;
	    rel_coord[2]=((i/9)%3)*4-4-(p2n & 4?2:0)+1;
	    int c_hash = coord_hash(rel_coord);
	    int idx=hash_lut[t][c_hash];
	    if(idx>=0) interac_list[idx]=pc;
	  }
	}
	break;
      }
    default:
      break;
    }
  }

  Matrix<Real_t>& ClassMat(int l, Mat_Type type, size_t indx){
    size_t indx0=InteracClass(type, indx);
    return mat->Mat(l, type, indx0);
  }

  Permutation<Real_t>& Perm_R(int l, Mat_Type type, size_t indx){
    size_t indx0=InteracClass(type, indx);
    Matrix     <Real_t>& M0      =mat->Mat   (l, type, indx0);
    Permutation<Real_t>& row_perm=mat->Perm_R(l, type, indx );
    if(M0.Dim(0)==0 || M0.Dim(1)==0) return row_perm;
    if(row_perm.Dim()==0){
      std::vector<Perm_Type> p_list=PermutList(type, indx);
      for(int i=0;i<l;i++) p_list.push_back(Scaling);
      Permutation<Real_t> row_perm_=Permutation<Real_t>(M0.Dim(0));
      for(int i=0;i<C_Perm;i++){
	Permutation<Real_t>& pr=mat->Perm(type, R_Perm + i);
	if(!pr.Dim()) row_perm_=Permutation<Real_t>(0);
      }
      if(row_perm_.Dim()>0)
	for(int i=p_list.size()-1; i>=0; i--){
	  assert(type!=V_Type);
	  Permutation<Real_t>& pr=mat->Perm(type, R_Perm + p_list[i]);
	  row_perm_=pr.Transpose()*row_perm_;
	}
      row_perm=row_perm_;
    }
    return row_perm;
  }

  Permutation<Real_t>& Perm_C(int l, Mat_Type type, size_t indx){
    size_t indx0=InteracClass(type, indx);
    Matrix     <Real_t>& M0      =mat->Mat   (l, type, indx0);
    Permutation<Real_t>& col_perm=mat->Perm_C(l, type, indx );
    if(M0.Dim(0)==0 || M0.Dim(1)==0) return col_perm;
    if(col_perm.Dim()==0){
      std::vector<Perm_Type> p_list=PermutList(type, indx);
      for(int i=0;i<l;i++) p_list.push_back(Scaling);
      Permutation<Real_t> col_perm_=Permutation<Real_t>(M0.Dim(1));
      for(int i=0;i<C_Perm;i++){
	Permutation<Real_t>& pc=mat->Perm(type, C_Perm + i);
	if(!pc.Dim()) col_perm_=Permutation<Real_t>(0);
      }
      if(col_perm_.Dim()>0)
	for(int i=p_list.size()-1; i>=0; i--){
	  assert(type!=V_Type);
	  Permutation<Real_t>& pc=mat->Perm(type, C_Perm + p_list[i]);
	  col_perm_=col_perm_*pc;
	}
      col_perm=col_perm_;
    }
    return col_perm;
  }

  void InitList(int max_r, int min_r, int step, Mat_Type t){
    const int max_hash = 2000;
    int n1 = (max_r*2)/step+1;
    int n2 = (min_r*2)/step-1;
    size_t count=n1*n1*n1-(min_r>0?n2*n2*n2:0);
    Matrix<int>& M=rel_coord[t];
    M.Resize(count,3);
    hash_lut[t].assign(max_hash, -1);
    std::vector<int> class_size_hash(max_hash, 0);
    std::vector<int> class_disp_hash(max_hash, 0);
    for(int k=-max_r;k<=max_r;k+=step)
      for(int j=-max_r;j<=max_r;j+=step)
	for(int i=-max_r;i<=max_r;i+=step)
	  if(abs(i)>=min_r || abs(j)>=min_r || abs(k) >= min_r){
	    int c[3]={i,j,k};
	    class_size_hash[class_hash(c)]++;
	  }
    scan(&class_size_hash[0], &class_disp_hash[0], max_hash);
    size_t count_=0;
    for(int k=-max_r;k<=max_r;k+=step)
      for(int j=-max_r;j<=max_r;j+=step)
	for(int i=-max_r;i<=max_r;i+=step)
	  if(abs(i)>=min_r || abs(j)>=min_r || abs(k) >= min_r){
	    int c[3]={i,j,k};
	    int& idx=class_disp_hash[class_hash(c)];
	    for(size_t l=0;l<3;l++) M[idx][l]=c[l];
	    hash_lut[t][coord_hash(c)]=idx;
	    count_++;
	    idx++;
	  }
    assert(count_==count);
    interac_class[t].resize(count);
    perm_list[t].resize(count);
    for(size_t j=0;j<count;j++){
      if(M[j][0]<0) perm_list[t][j].push_back(ReflecX);
      if(M[j][1]<0) perm_list[t][j].push_back(ReflecY);
      if(M[j][2]<0) perm_list[t][j].push_back(ReflecZ);
      int coord[3];
      coord[0]=abs(M[j][0]);
      coord[1]=abs(M[j][1]);
      coord[2]=abs(M[j][2]);
      if(coord[1]>coord[0] && coord[1]>coord[2]){
	perm_list[t][j].push_back(SwapXY);
	int tmp=coord[0]; coord[0]=coord[1]; coord[1]=tmp;
      }
      if(coord[0]>coord[2]){
	perm_list[t][j].push_back(SwapXZ);
	int tmp=coord[0]; coord[0]=coord[2]; coord[2]=tmp;
      }
      if(coord[0]>coord[1]){
	perm_list[t][j].push_back(SwapXY);
	int tmp=coord[0]; coord[0]=coord[1]; coord[1]=tmp;
      }
      assert(coord[0]<=coord[1] && coord[1]<=coord[2]);
      int c_hash = coord_hash(&coord[0]);
      interac_class[t][j]=hash_lut[t][c_hash];
    }
  }

  int coord_hash(int* c){
    const int n=5;
    return ( (c[2]+n) * (2*n) + (c[1]+n) ) *(2*n) + (c[0]+n);
  }

  int class_hash(int* c_){
    int c[3]={abs(c_[0]), abs(c_[1]), abs(c_[2])};
    if(c[1]>c[0] && c[1]>c[2])
      {int tmp=c[0]; c[0]=c[1]; c[1]=tmp;}
    if(c[0]>c[2])
      {int tmp=c[0]; c[0]=c[2]; c[2]=tmp;}
    if(c[0]>c[1])
      {int tmp=c[0]; c[0]=c[1]; c[1]=tmp;}
    assert(c[0]<=c[1] && c[1]<=c[2]);
    return coord_hash(&c[0]);
  }

};

}//end namespace

#endif //_PVFMM_INTERAC_LIST_HPP_

