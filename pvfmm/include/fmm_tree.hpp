#include <mpi_tree.hpp>

#ifndef _PVFMM_FMM_TREE_HPP_
#define _PVFMM_FMM_TREE_HPP_

namespace pvfmm{

template <class FMM_Mat_t>
class FMM_Tree: public MPI_Tree<typename FMM_Mat_t::FMMNode_t>{

 public:

  typedef typename FMM_Mat_t::FMMNode_t Node_t;

  std::vector<Matrix<Real_t> > node_data_buff;
  pvfmm::Matrix<Node_t*> node_interac_lst;
  InteracList<Node_t> interac_list;
  FMM_Mat_t* fmm_mat;
  BoundaryType bndry;
  std::vector<Matrix<char> > precomp_lst;
  std::vector<SetupData<Real_t> > setup_data;
  std::vector<Vector<Real_t> > upwd_check_surf;
  std::vector<Vector<Real_t> > upwd_equiv_surf;
  std::vector<Vector<Real_t> > dnwd_check_surf;
  std::vector<Vector<Real_t> > dnwd_equiv_surf;

  FMM_Tree(): MPI_Tree<Node_t>(), fmm_mat(NULL), bndry(FreeSpace) { };

  virtual ~FMM_Tree(){}

  virtual void Initialize(typename Node_t::NodeData* data_) ;

  void InitFMM_Tree(bool refine, BoundaryType bndry=FreeSpace);

  void SetupFMM(FMM_Mat_t* fmm_mat_);

  void RunFMM();

  void ClearFMMData();

  void BuildInteracLists();

  void UpwardPass();

  void DownwardPass();

};

}//end namespace

#include <fmm_tree.txx>

#endif //_PVFMM_FMM_TREE_HPP_

