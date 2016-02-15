namespace pvfmm{

template <class Node_t>
void InteracList<Node_t>::Initialize(unsigned int dim_, PrecompMat<Real_t>* mat_){
  #ifdef PVFMM_NO_SYMMETRIES
  use_symmetries=false;
  #else
  use_symmetries=true;
  #endif

  dim=dim_;
  assert(dim==3); //Only supporting 3D for now.
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

template <class Node_t>
size_t InteracList<Node_t>::ListCount(Mat_Type t){
  return rel_coord[t].Dim(0);
}

template <class Node_t>
int* InteracList<Node_t>::RelativeCoord(Mat_Type t, size_t i){
  return rel_coord[t][i];
}

template <class Node_t>
  size_t InteracList<Node_t>::InteracClass(Mat_Type t, size_t i){
  return interac_class[t][i];
}
template <class Node_t>
std::vector<Perm_Type>& InteracList<Node_t>::PermutList(Mat_Type t, size_t i){
  return perm_list[t][i];
}

template <class Node_t>
void InteracList<Node_t>::BuildList(Node_t* n, Mat_Type t){
  Vector<Node_t*>& interac_list=n->interac_list[t];
  if(interac_list.Dim()!=ListCount(t)) interac_list.ReInit(ListCount(t));
  interac_list.SetZero();

  static const int n_collg=pvfmm::pow<unsigned int>(3,dim);
  static const int n_child=pvfmm::pow<unsigned int>(2,dim);
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
        Node_t* chld=(Node_t*)n->Child(j);
        if(idx>=0 && !chld->IsGhost()) interac_list[idx]=chld;
      }
      break;
    }
    case D2D_Type:
    {
      if(n->IsGhost() || n->Parent()==NULL) return;
      Node_t* p=(Node_t*)n->Parent();
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
      Node_t* p=(Node_t*)n->Parent();
      int p2n=n->Path2Node();
      for(int i=0;i<n_collg;i++){
        Node_t* pc=(Node_t*)p->Colleague(i);
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
        Node_t* col=(Node_t*)n->Colleague(i);
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
        Node_t* col=(Node_t*)n->Colleague(i);
        if(col!=NULL && !col->IsLeaf()){
          for(int j=0;j<n_child;j++){
            rel_coord[0]=( i %3)*4-4+(j & 1?2:0)-1;
            rel_coord[1]=((i/3)%3)*4-4+(j & 2?2:0)-1;
            rel_coord[2]=((i/9)%3)*4-4+(j & 4?2:0)-1;
            int c_hash = coord_hash(rel_coord);
            int idx=hash_lut[t][c_hash];
            if(idx>=0){
              assert(col->Child(j)->IsLeaf()); //2:1 balanced
              interac_list[idx]=(Node_t*)col->Child(j);
            }
          }
        }
      }
      break;
    }
    case V_Type:
    {
      if(n->IsGhost() || n->Parent()==NULL) return;
      Node_t* p=(Node_t*)n->Parent();
      int p2n=n->Path2Node();
      for(int i=0;i<n_collg;i++){
        Node_t* pc=(Node_t*)p->Colleague(i);
        if(pc!=NULL?!pc->IsLeaf():0){
          for(int j=0;j<n_child;j++){
            rel_coord[0]=( i   %3)*2-2+(j & 1?1:0)-(p2n & 1?1:0);
            rel_coord[1]=((i/3)%3)*2-2+(j & 2?1:0)-(p2n & 2?1:0);
            rel_coord[2]=((i/9)%3)*2-2+(j & 4?1:0)-(p2n & 4?1:0);
            int c_hash = coord_hash(rel_coord);
            int idx=hash_lut[t][c_hash];
            if(idx>=0) interac_list[idx]=(Node_t*)pc->Child(j);
          }
        }
      }
      break;
    }
    case V1_Type:
    {
      if(n->IsGhost() || n->IsLeaf()) return;
      for(int i=0;i<n_collg;i++){
        Node_t* col=(Node_t*)n->Colleague(i);
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
        Node_t* col=(Node_t*)n->Colleague(i);
        if(col!=NULL && !col->IsLeaf()){
          for(int j=0;j<n_child;j++){
            rel_coord[0]=( i %3)*4-4+(j & 1?2:0)-1;
            rel_coord[1]=((i/3)%3)*4-4+(j & 2?2:0)-1;
            rel_coord[2]=((i/9)%3)*4-4+(j & 4?2:0)-1;
            int c_hash = coord_hash(rel_coord);
            int idx=hash_lut[t][c_hash];
            if(idx>=0) interac_list[idx]=(Node_t*)col->Child(j);
          }
        }
      }
      break;
    }
    case X_Type:
    {
      if(n->IsGhost() || n->Parent()==NULL) return;
      Node_t* p=(Node_t*)n->Parent();
      int p2n=n->Path2Node();
      for(int i=0;i<n_collg;i++){
        Node_t* pc=(Node_t*)p->Colleague(i);
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

template <class Node_t>
Matrix<Real_t>& InteracList<Node_t>::ClassMat(int l, Mat_Type type, size_t indx){
  size_t indx0=InteracClass(type, indx);
  return mat->Mat(l, type, indx0);
}


}//end namespace
