namespace pvfmm{

template <class TreeNode>
void MPI_Tree<TreeNode>::SetColleagues(BoundaryType bndry, TreeNode* node){
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

}//end namespace
