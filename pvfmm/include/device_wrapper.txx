/**
 * \file device_wrapper.txx
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 6-5-2013
 * \brief This file contains implementation of DeviceWrapper.
 *
 * Modified:
 *   editor Chenhan D. Yu
 *   date Juan-28-2014
 *   Add Cuda support. Error handle is available if needed.
 */

#include <omp.h>
#include <cassert>
#include <cstdlib>

namespace pvfmm{

namespace DeviceWrapper{
  #define ALLOC alloc_if(1) free_if(0)
  #define FREE alloc_if(0) free_if(1)
  #define REUSE alloc_if(0) free_if(0)

  // Wrapper functions

  inline void* host_malloc(size_t size){
    if(!size) return NULL;
    return malloc(size);
  }

  inline void host_free(void* p){
    return free(p);
  }

  inline uintptr_t alloc_device(char* dev_handle, size_t len){
    UNUSED(len);
    uintptr_t dev_ptr=(uintptr_t)NULL;
    {dev_ptr=(uintptr_t)dev_handle;}
    return dev_ptr;
  }

  inline void free_device(char* dev_handle, uintptr_t dev_ptr){
    UNUSED(dev_handle);
    UNUSED(dev_ptr);
  }

  template <int SYNC>
  inline int host2device(char* host_ptr, char* dev_handle, uintptr_t dev_ptr, size_t len){
    int lock_idx=-1;
    return lock_idx;
  }

  template <int SYNC>
  inline int device2host(char* dev_handle, uintptr_t dev_ptr, char* host_ptr, size_t len){
    int lock_idx=-1;
    UNUSED(dev_handle);
    UNUSED(host_ptr);
    UNUSED(dev_ptr);
    UNUSED(len);
    return lock_idx;
  }

  inline void wait(int lock_idx){
    UNUSED(lock_idx);
  }

}//end namespace
}//end namespace
