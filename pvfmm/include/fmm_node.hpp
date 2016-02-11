#include <vector>
#include <cstdlib>

#include <pvfmm_common.hpp>
#include <mpi_node.hpp>
#include <fmm_pts.hpp>

#ifndef _PVFMM_FMM_NODE_HPP_
#define _PVFMM_FMM_NODE_HPP_

namespace pvfmm{

/**
 * \brief Base class for node of FMM_Node.
 */
class FMM_Node: public MPI_Node {

 public:

  class NodeData: public MPI_Node::NodeData{

    public:

     Vector<Real_t> src_coord; //Point sources.
     Vector<Real_t> src_value;

     Vector<Real_t> surf_coord; //Surface sources.
     Vector<Real_t> surf_value;

     Vector<Real_t> trg_coord; //Target coordinates.
     Vector<Real_t> trg_value;
  };

  FMM_Node(){
    MPI_Node();
    fmm_data=NULL;
  }

  virtual ~FMM_Node();

  virtual void Initialize(TreeNode* parent_, int path2node_, TreeNode::NodeData*) ;

  virtual void NodeDataVec(std::vector<Vector<Real_t>*>& coord,
                           std::vector<Vector<Real_t>*>& value,
                           std::vector<Vector<size_t>*>& scatter){
    MPI_Node::NodeDataVec(coord, value, scatter);
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

  virtual void ClearData();

  void ClearFMMData();

  FMM_Data<Real_t>*& FMMData(){return fmm_data;}

  virtual TreeNode* NewNode(TreeNode* n_=NULL);

  virtual void Subdivide() ;

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
