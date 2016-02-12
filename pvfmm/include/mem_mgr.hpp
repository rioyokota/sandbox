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
    static inline uintptr_t ID();
    static inline bool IsPOD();
};

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

    MemoryManager(size_t N);

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
