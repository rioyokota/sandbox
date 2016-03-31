
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

  ~MemoryManager(){
    MemNode* n_dummy=&node_buff[n_dummy_indx-1];
    MemNode* n=&node_buff[n_dummy->next-1];
    if(!n->free || n->size!=buff_size ||
       node_stack.size()!=node_buff.size()-2){
      std::cout<<"\nWarning: memory leak detected.\n";
    }
    omp_destroy_lock(&omp_lock);

    { // free buff
      assert(buff);
      std::free(buff-((uint16_t*)buff)[-1]);
    }
  }

  static inline MemHead* GetMemHead(void* p){
    static uintptr_t alignment=MEM_ALIGN-1;
    static uintptr_t header_size=(uintptr_t)(sizeof(MemoryManager::MemHead)+alignment) & ~(uintptr_t)alignment;
    return (MemHead*)(((char*)p)-header_size);
  }

  void* malloc(const size_t n_elem=1, const size_t type_size=sizeof(char)) const{
    if(!n_elem) return NULL;
    static uintptr_t alignment=MEM_ALIGN-1;
    static uintptr_t header_size=(uintptr_t)(sizeof(MemHead)+alignment) & ~(uintptr_t)alignment;

    size_t size=n_elem*type_size+header_size;
    size=(uintptr_t)(size+alignment) & ~(uintptr_t)alignment;
    char* base=NULL;

    omp_set_lock(&omp_lock);
    std::multimap<size_t, size_t>::iterator it=free_map.lower_bound(size);
    size_t n_indx=(it!=free_map.end()?it->second:0);
    omp_unset_lock(&omp_lock);
    if(!base){ // Use system malloc
      size+=2+alignment;
      char* p = (char*)std::malloc(size);
      base = (char*)((uintptr_t)(p+2+alignment) & ~(uintptr_t)alignment);
      ((uint16_t*)base)[-1] = (uint16_t)(base-p);
    }
    MemHead* mem_head=(MemHead*)base;
    { // Set mem_head
      mem_head->n_indx=n_indx;
      mem_head->n_elem=n_elem;
      mem_head->type_size=type_size;
    }
    return (void*)(base+header_size);
  }

  void free(void* p) const{
    if(!p) return;
    char* base=(char*)((char*)p-MEM_ALIGN);
    char* p_=(char*)((uintptr_t)base-((uint16_t*)base)[-1]);
    return std::free(p_);
  }

private:

  struct MemNode{
    bool free;
    size_t size;
    char* mem_ptr;
    size_t prev, next;
    std::multimap<size_t, size_t>::iterator it;
  };

  inline size_t new_node() const{
    if(node_stack.empty()){
      node_buff.resize(node_buff.size()+1);
      node_stack.push(node_buff.size());
    }
    size_t indx=node_stack.top();
    node_stack.pop();
    assert(indx);
    return indx;
  }

  char* buff;
  size_t buff_size;
  size_t n_dummy_indx;

  mutable std::vector<MemNode> node_buff;
  mutable std::stack<size_t> node_stack;
  mutable std::multimap<size_t, size_t> free_map;
  mutable omp_lock_t omp_lock;
};

MemoryManager glbMemMgr(16*1024*1024*sizeof(Real_t));

inline uintptr_t align_ptr(uintptr_t ptr){
  static uintptr_t     ALIGN_MINUS_ONE=MEM_ALIGN-1;
  static uintptr_t NOT_ALIGN_MINUS_ONE=~ALIGN_MINUS_ONE;
  return ((ptr+ALIGN_MINUS_ONE) & NOT_ALIGN_MINUS_ONE);
}
  
template <class T>
inline T* aligned_new(size_t n_elem=1, const MemoryManager* mem_mgr=&glbMemMgr) {
  if(!n_elem) return NULL;
  T* A=(T*)mem_mgr->malloc(n_elem, sizeof(T));
  assert(A);
  return A;
}

template <class T>
inline void aligned_delete(T* A, const MemoryManager* mem_mgr=&glbMemMgr){
  if (!A) return;
  mem_mgr->free(A);
}

}//end namespace
}//end namespace

#endif //_PVFMM_MEM_MGR_HPP_
