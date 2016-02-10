#include <vector>
#include <cassert>
#include <cstdlib>
#include <stdint.h>

#include <pvfmm_common.hpp>
#include <tree_node.hpp>
#include <mortonid.hpp>

#ifndef _PVFMM_MPI_NODE_HPP_
#define _PVFMM_MPI_NODE_HPP_

namespace pvfmm{

class MPI_Node: public TreeNode{

 protected:

  MPI_Node * colleague[COLLEAGUE_COUNT];

 public:

  Vector<Real_t> pt_coord;
  Vector<Real_t> pt_value;
  Vector<size_t> pt_scatter;

  MPI_Node(): TreeNode() {}

  virtual ~MPI_Node();

  virtual void Initialize(TreeNode* parent_, int path2node_, TreeNode::NodeData*) ;

  virtual void NodeDataVec(std::vector<Vector<Real_t>*>& coord,
                           std::vector<Vector<Real_t>*>& value,
                           std::vector<Vector<size_t>*>& scatter){
    coord  .push_back(&pt_coord  );
    value  .push_back(&pt_value  );
    scatter.push_back(&pt_scatter);
  }

  virtual void ClearData();

  MPI_Node * Colleague(int index){return colleague[index];}

  void SetColleague(MPI_Node * node_, int index){colleague[index]=node_;}

  virtual long long& NodeCost(){return weight;}

  Real_t* Coord(){assert(coord!=NULL); return coord;}

  bool IsGhost(){return ghost;}

  void SetGhost(bool x){ghost=x;}

  inline MortonId GetMortonId();

  inline void SetCoord(MortonId& mid);

  virtual TreeNode* NewNode(TreeNode* n_=NULL);

  virtual void Subdivide();

  virtual void ReadVal(std::vector<Real_t> x,std::vector<Real_t> y, std::vector<Real_t> z, Real_t* val, bool show_ghost=true);

};

}//end namespace

#include <mpi_node.txx>

#endif //_PVFMM_MPI_NODE_HPP_
