#ifndef _PVFMM_FMM_KERNEL_HPP_
#define _PVFMM_FMM_KERNEL_HPP_

#include <precomp_mat.hpp>
#include <intrin_wrapper.hpp>

namespace pvfmm{

template <class T>
struct Kernel{
  public:

  typedef void (*Ker_t)(T* r_src, int src_cnt, T* v_src, int dof,
                        T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr);

  typedef void (*VolPoten)(const T* coord, int n, T* out);

  Kernel(Ker_t poten, const char* name, int dim_, std::pair<int,int> k_dim);

  void Initialize(bool verbose=false) const;

  void BuildMatrix(T* r_src, int src_cnt,
                   T* r_trg, int trg_cnt, T* k_out) const;

  int dim;
  int ker_dim[2];
  std::string ker_name;
  Ker_t ker_poten;

  mutable bool init;
  mutable bool scale_invar;
  mutable Vector<T> src_scal;
  mutable Vector<T> trg_scal;
  mutable Vector<Permutation<T> > perm_vec;

  mutable const Kernel<T>* k_s2m;
  mutable const Kernel<T>* k_s2l;
  mutable const Kernel<T>* k_s2t;
  mutable const Kernel<T>* k_m2m;
  mutable const Kernel<T>* k_m2l;
  mutable const Kernel<T>* k_m2t;
  mutable const Kernel<T>* k_l2l;
  mutable const Kernel<T>* k_l2t;
  mutable VolPoten vol_poten;

  private:

  Kernel();

};

template<typename T, void (*A)(T*, int, T*, int, T*, int, T*, mem::MemoryManager* mem_mgr)>
Kernel<T> BuildKernel(const char* name, int dim, std::pair<int,int> k_dim,
    const Kernel<T>* k_s2m=NULL, const Kernel<T>* k_s2l=NULL, const Kernel<T>* k_s2t=NULL,
    const Kernel<T>* k_m2m=NULL, const Kernel<T>* k_m2l=NULL, const Kernel<T>* k_m2t=NULL,
    const Kernel<T>* k_l2l=NULL, const Kernel<T>* k_l2t=NULL, typename Kernel<T>::VolPoten vol_poten=NULL){
  Kernel<T> K(A, name, dim, k_dim);
  K.k_s2m=k_s2m;
  K.k_s2l=k_s2l;
  K.k_s2t=k_s2t;
  K.k_m2m=k_m2m;
  K.k_m2l=k_m2l;
  K.k_m2t=k_m2t;
  K.k_l2l=k_l2l;
  K.k_l2t=k_l2t;
  K.vol_poten=vol_poten;
  return K;
}

template<class T>
struct LaplaceKernel{
  inline static const Kernel<T>& gradient();
};

}//end namespace

#include <cheb_utils.hpp>
#include <kernel.txx>

#endif //_PVFMM_FMM_KERNEL_HPP_

