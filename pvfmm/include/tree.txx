namespace pvfmm{

template <class TreeNode>
Tree<TreeNode>::~Tree(){
  if(RootNode()!=NULL){
    mem::aligned_delete(root_node);
  }
}

template <class TreeNode>
void Tree<TreeNode>::Initialize(typename TreeNode::NodeData* init_data_){
  dim=init_data_->dim;
  max_depth=init_data_->max_depth;
  if(max_depth>MAX_DEPTH) max_depth=MAX_DEPTH;

  if(root_node) mem::aligned_delete(root_node);
  root_node=mem::aligned_new<TreeNode>();
  root_node->Initialize(NULL,0,init_data_);
}

template <class TreeNode>
TreeNode* Tree<TreeNode>::PreorderFirst(){
  return root_node;
}

template <class TreeNode>
TreeNode* Tree<TreeNode>::PreorderNxt(TreeNode* curr_node){
  assert(curr_node!=NULL);

  int n=(1UL<<dim);
  if(!curr_node->IsLeaf())
    for(int i=0;i<n;i++)
      if(curr_node->Child(i)!=NULL)
        return (TreeNode*)curr_node->Child(i);

  TreeNode* node=curr_node;
  while(true){
    int i=node->Path2Node()+1;
    node=(TreeNode*)node->Parent();
    if(node==NULL) return NULL;

    for(;i<n;i++)
      if(node->Child(i)!=NULL)
        return (TreeNode*)node->Child(i);
  }
}

template <class TreeNode>
TreeNode* Tree<TreeNode>::PostorderFirst(){
  TreeNode* node=root_node;

  int n=(1UL<<dim);
  while(true){
    if(node->IsLeaf()) return node;
    for(int i=0;i<n;i++)
      if(node->Child(i)!=NULL){
        node=(TreeNode*)node->Child(i);
        break;
      }
  }
}

template <class TreeNode>
TreeNode* Tree<TreeNode>::PostorderNxt(TreeNode* curr_node){
  assert(curr_node!=NULL);
  TreeNode* node=curr_node;

  int j=node->Path2Node()+1;
  node=(TreeNode*)node->Parent();
  if(node==NULL) return NULL;

  int n=(1UL<<dim);
  for(;j<n;j++){
    if(node->Child(j)!=NULL){
      node=(TreeNode*)node->Child(j);
      while(true){
        if(node->IsLeaf()) return node;
        for(int i=0;i<n;i++)
          if(node->Child(i)!=NULL){
            node=(TreeNode*)node->Child(i);
            break;
          }
      }
    }
  }

  return node;
}

template <class TreeNode>
std::vector<TreeNode*>& Tree<TreeNode>::GetNodeList(){
  if(root_node->GetStatus() & 1){
    node_lst.clear();
    TreeNode* n=this->PreorderFirst();
    while(n!=NULL){
      int& status=n->GetStatus();
      status=(status & (~(int)1));
      node_lst.push_back(n);
      n=this->PreorderNxt(n);
    }
  }
  return node_lst;
}

}//end namespace
