#include <cmath>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <stdint.h>

#include <pvfmm_common.hpp>
#include <matrix.hpp>
#include <mem_mgr.hpp>
#include <mortonid.hpp>
#include <vector.hpp>

#ifndef _PVFMM_MPI_NODE_HPP_
#define _PVFMM_MPI_NODE_HPP_

namespace pvfmm{

class MPI_Node {

 public:

  int dim;
  int depth;
  int max_depth;
  int path2node;
  MPI_Node* parent;
  MPI_Node** child;
  int status;

  bool ghost;
  size_t max_pts;
  size_t node_id;
  long long weight;

  Real_t coord[COORD_DIM];
  MPI_Node * colleague[COLLEAGUE_COUNT];

  Vector<Real_t> pt_coord;
  Vector<Real_t> pt_value;
  Vector<size_t> pt_scatter;

  class NodeData{
   public:
     virtual ~NodeData(){};
     virtual void Clear(){}
     int max_depth;
     int dim;
     size_t max_pts;
     Vector<Real_t> coord;
     Vector<Real_t> value;
  };

  MPI_Node(): dim(0), depth(0), max_depth(MAX_DEPTH), parent(NULL), child(NULL), status(1),
              ghost(false), weight(1) {}

  virtual ~MPI_Node() {
    if(!child) return;
    int n=(1UL<<dim);
    for(int i=0;i<n;i++){
      if(child[i]!=NULL)
	mem::aligned_delete(child[i]);
    }
    mem::aligned_delete(child);
    child=NULL;
  }

};

}//end namespace

#endif //_PVFMM_MPI_NODE_HPP_
