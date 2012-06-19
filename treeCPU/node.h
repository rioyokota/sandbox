#include <cassert>

struct Node{
  static const int NLEAF = 65;
  int np;
  int ifirst;
  Node() : np(0), ifirst(-1) {}

  static Node *&node_ptr(){
    static Node *ptr = NULL;
    return ptr;
  }
  static int &node_count(){
    static int count = 0;
    return count;
  }
  static int &node_limit(){
    static int limit = 0;
    return limit;
  }

  bool is_leaf() const{
    return (np < NLEAF);
  }
  void insert(
      int i,
      unsigned long long key[],
      int rshift = 60)
  {
    assert(rshift >= 0);
    np++;
    if(np < NLEAF){ // is a leaf
      if(np==1){
        assert(ifirst == -1);
        ifirst = i;
      }
    }
    else if(np == NLEAF){ // has become a node
      int pfirst = ifirst;
      ifirst = node_count();
      node_count() += 8;
      assert(node_count() <= node_limit());
      Node *child = node_ptr() + ifirst;
      for(int j=0; j<np; j++){
        int oct = (key[pfirst + j] >> rshift) & 7;
        child[oct].insert(pfirst + j, key, rshift-3);
      }
    }
    else if(np > NLEAF){ // is a node
      int oct = (key[i] >> rshift) & 7;
      Node *child = node_ptr() + ifirst;
      child[oct].insert(i, key, rshift-3);
    }
  }
};
