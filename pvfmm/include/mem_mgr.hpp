#include <omp.h>
#include <cstdlib>
#include <stdint.h>
#include <cassert>
#include <vector>
#include <stack>
#include <map>

#include <pvfmm_common.hpp>

#ifndef _PVFMM_MEM_MGR_HPP_
#define _PVFMM_MEM_MGR_HPP_

namespace pvfmm{
namespace mem{

template <class T>
class TypeTraits{
public:
  static inline uintptr_t ID(){
    return (uintptr_t)&ID;
  }
  static inline bool IsPOD(){
    return false;
  }
};

#define PVFMMDefinePOD(type) template<> bool inline TypeTraits<type>::IsPOD(){return true;};
PVFMMDefinePOD(char);
PVFMMDefinePOD(float);
PVFMMDefinePOD(double);
PVFMMDefinePOD(int);
PVFMMDefinePOD(long long);
PVFMMDefinePOD(unsigned long);
PVFMMDefinePOD(char*);
PVFMMDefinePOD(float*);
PVFMMDefinePOD(double*);
#undef PVFMMDefinePOD

class MemoryManager{
public:
  static const char init_mem_val=42;

  struct MemHead{
    size_t n_indx;
    size_t n_elem;
    uintptr_t type_id;
    uintptr_t type_size;
    unsigned char check_sum;
  };

  MemoryManager(size_t N){
    buff_size=N;
    { // Allocate buff
      assert(MEM_ALIGN <= 0x8000);
      size_t alignment=MEM_ALIGN-1;
      char* base_ptr=(char*)std::malloc(N+2+alignment); assert(base_ptr);
      buff=(char*)((uintptr_t)(base_ptr+2+alignment) & ~(uintptr_t)alignment);
      ((uint16_t*)buff)[-1] = (uint16_t)(buff-base_ptr);
    }
    { // Initialize to init_mem_val
#ifndef NDEBUG
#pragma omp parallel for
      for(size_t i=0;i<buff_size;i++){
	buff[i]=init_mem_val;
      }
#endif
    }
    n_dummy_indx=new_node();
    size_t n_indx=new_node();
    MemNode& n_dummy=node_buff[n_dummy_indx-1];
    MemNode& n=node_buff[n_indx-1];

    n_dummy.size=0;
    n_dummy.free=false;
    n_dummy.prev=0;
    n_dummy.next=n_indx;
    n_dummy.mem_ptr=&buff[0];
    assert(n_indx);

    n.size=N;
    n.free=true;
    n.prev=n_dummy_indx;
    n.next=0;
    n.mem_ptr=&buff[0];
    n.it=free_map.insert(std::make_pair(N,n_indx));

    omp_init_lock(&omp_lock);
  }

  ~MemoryManager();

  static inline MemHead* GetMemHead(void* p);

  void* malloc(const size_t n_elem=1, const size_t type_size=sizeof(char)) const;

  void free(void* p) const;

  void print() const;

  static void test();

  void Check() const;

private:

  MemoryManager();

  MemoryManager(const MemoryManager& m);

  struct MemNode{
    bool free;
    size_t size;
    char* mem_ptr;
    size_t prev, next;
    std::multimap<size_t, size_t>::iterator it;
  };

  inline size_t new_node() const;

  inline void delete_node(size_t indx) const;

  char* buff;
  size_t buff_size;
  size_t n_dummy_indx;

  mutable std::vector<MemNode> node_buff;
  mutable std::stack<size_t> node_stack;
  mutable std::multimap<size_t, size_t> free_map;
  mutable omp_lock_t omp_lock;
};

  extern MemoryManager glbMemMgr;

  inline uintptr_t align_ptr(uintptr_t ptr){
    static uintptr_t     ALIGN_MINUS_ONE=MEM_ALIGN-1;
    static uintptr_t NOT_ALIGN_MINUS_ONE=~ALIGN_MINUS_ONE;
    return ((ptr+ALIGN_MINUS_ONE) & NOT_ALIGN_MINUS_ONE);
  }

  template <class T>
  inline T* aligned_new(size_t n_elem=1, const MemoryManager* mem_mgr=&glbMemMgr);

  template <class T>
  inline void aligned_delete(T* A, const MemoryManager* mem_mgr=&glbMemMgr);

  inline void * memcopy(void * destination, const void * source, size_t num);

}//end namespace
}//end namespace

#include <mem_mgr.txx>

#endif //_PVFMM_MEM_MGR_HPP_
