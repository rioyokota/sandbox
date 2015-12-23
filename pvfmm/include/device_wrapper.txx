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

  inline void wait(int lock_idx){
    UNUSED(lock_idx);
  }

}//end namespace
}//end namespace
