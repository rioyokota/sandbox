/**
 * \file fmm_node.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 12-11-2010
 * \brief This file contains the definition of the FMM_Node class.
 */

#include <vector>
#include <cstdlib>

#include <pvfmm_common.hpp>
#include <tree_node.hpp>
#include <mpi_node.hpp>
#include <fmm_pts.hpp>
#include <vector.hpp>

#ifndef _PVFMM_FMM_NODE_HPP_
#define _PVFMM_FMM_NODE_HPP_

namespace pvfmm{

/**
 * \brief Base class for node of FMM_Node.
 */
template <class Node>
class FMM_Node: public Node{

 public:

  typedef Node Node_t; //Type of base node class

  /**
   * \brief Base class for node data. Contains initialization data for the node.
   */
  class NodeData: public Node_t::NodeData{

    public:

     Vector<Real_t> src_coord; //Point sources.
     Vector<Real_t> src_value;

     Vector<Real_t> surf_coord; //Surface sources.
     Vector<Real_t> surf_value;

     Vector<Real_t> trg_coord; //Target coordinates.
     Vector<Real_t> trg_value;
  };

  /**
   * \brief Constructor.
   */
  FMM_Node(){
    Node_t();
    fmm_data=NULL;
  }

  /**
   * /brief Virtual destructor.
   */
  virtual ~FMM_Node();

  /**
   * \brief Initialize the node with relevant data.
   */
  virtual void Initialize(TreeNode* parent_, int path2node_, TreeNode::NodeData*) ;

  /**
   * \brief Returns list of coordinate and value vectors which need to be
   * sorted and partitioned across processes and the scatter index is
   * saved.
   */
  virtual void NodeDataVec(std::vector<Vector<Real_t>*>& coord,
                           std::vector<Vector<Real_t>*>& value,
                           std::vector<Vector<size_t>*>& scatter){
    Node::NodeDataVec(coord, value, scatter);
    coord  .push_back(&src_coord  );
    value  .push_back(&src_value  );
    scatter.push_back(&src_scatter);

    coord  .push_back(&surf_coord  );
    value  .push_back(&surf_value  );
    scatter.push_back(&surf_scatter);

    coord  .push_back(&trg_coord  );
    value  .push_back(&trg_value  );
    scatter.push_back(&trg_scatter);
  }

  /**
   * \brief Clear node data.
   */
  virtual void ClearData();

  /**
   * \brief Clear FMM specific node (multipole and local expansion) data
   */
  void ClearFMMData();

  /**
   * \brief Returns reference to fmm_data.
   */
  FMM_Data<Real_t>*& FMMData(){return fmm_data;}

  /**
   * \brief Allocate a new object of the same type (as the derived class) and
   * return a pointer to it type cast as (TreeNode*).
   */
  virtual TreeNode* NewNode(TreeNode* n_=NULL);

  /**
   * \brief Create child nodes and Initialize them.
   */
  virtual void Subdivide() ;

  /**
   * \brief Truncates the tree i.e. makes this a leaf node.
   */
  virtual void Truncate() ;

  Vector<Real_t> src_coord;  //Point sources.
  Vector<Real_t> src_value;
  Vector<size_t> src_scatter;

  Vector<Real_t> surf_coord; //Surface sources.
  Vector<Real_t> surf_value; //Normal and src strength.
  Vector<size_t> surf_scatter;

  Vector<Real_t> trg_coord;  //Target coordinates.
  Vector<Real_t> trg_value;
  Vector<size_t> trg_scatter;

  size_t pt_cnt[2]; // Number of source, target pts.
  Vector<FMM_Node*> interac_list[Type_Count];

 private:

  FMM_Data<Real_t>* fmm_data; //FMM specific data.
  Vector<char> pkd_data; //Temporary variable for storing packed data.
};

}//end namespace

#include <fmm_node.txx>

#endif //_PVFMM_FMM_NODE_HPP_
