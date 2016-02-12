#ifndef _PVFMM_MPI_TREE_HPP_
#define _PVFMM_MPI_TREE_HPP_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <set>
#include <sstream>
#include <stdint.h>
#include <string>
#include <vector>

#include <fmm_node.hpp>
#include <mem_mgr.hpp>
#include <mortonid.hpp>
#include <ompUtils.h>
#include <parUtils.h>
#include <profile.hpp>
#include <pvfmm_common.hpp>
#include <tree.hpp>

namespace pvfmm{

enum BoundaryType{
  FreeSpace,
  Periodic
};

template <class TreeNode>
class MPI_Tree: public Tree<TreeNode>{

 public:

  MPI_Tree(): Tree<TreeNode>() {}
  virtual ~MPI_Tree() {}
  virtual void Initialize(typename TreeNode::NodeData* data_);
  TreeNode* FindNode(MortonId& key, bool subdiv, TreeNode* start=NULL);
  void SetColleagues(BoundaryType bndry=FreeSpace, TreeNode* node=NULL) ;

 private:

  std::vector<MortonId> mins;

};

}//end namespace

#include <mpi_tree.txx>

#endif //_PVFMM_MPI_TREE_HPP_
