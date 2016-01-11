#include <vector>
#include <string>

#include <pvfmm_common.hpp>
#include <mortonid.hpp>
#include <tree.hpp>

#ifndef _PVFMM_MPI_TREE_HPP_
#define _PVFMM_MPI_TREE_HPP_

namespace pvfmm{

enum BoundaryType{
  FreeSpace,
  Periodic
};

/**
 * \brief Base class for distributed tree.
 */
template <class TreeNode>
class MPI_Tree: public Tree<TreeNode>{

 public:

  typedef TreeNode Node_t;
  typedef typename Node_t::Real_t Real_t;

  MPI_Tree(): Tree<Node_t>() {}
  virtual ~MPI_Tree() {}
  virtual void Initialize(typename Node_t::NodeData* data_);
  TreeNode* FindNode(MortonId& key, bool subdiv, TreeNode* start=NULL);
  void SetColleagues(BoundaryType bndry=FreeSpace, Node_t* node=NULL) ;

  template <class VTKReal>
  struct VTUData_t{
    typedef VTKReal VTKReal_t;

    // Point data
    std::vector<VTKReal_t> coord;
    std::vector<std::vector<VTKReal_t> > value;
    std::vector<std::string> name;

    // Cell data
    std::vector<int32_t> connect;
    std::vector<int32_t> offset;
    std::vector<uint8_t> types;
  };

 private:

  std::vector<MortonId> mins;

};

}//end namespace

#include <mpi_tree.txx>

#endif //_PVFMM_MPI_TREE_HPP_
