/**
 * \file device_wrapper.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 6-5-2013
 * \brief This file contains definition of DeviceWrapper.
 */

#include <cstdlib>
#include <stdint.h>

// Cuda Headers
#include <pvfmm_common.hpp>
#include <vector.hpp>

#ifndef _PVFMM_DEVICE_WRAPPER_HPP_
#define _PVFMM_DEVICE_WRAPPER_HPP_

namespace pvfmm{

namespace DeviceWrapper{

  void* host_malloc(size_t size);

  void host_free(void*);

  uintptr_t alloc_device(char* dev_handle, size_t len);

  void free_device(char* dev_handle, uintptr_t dev_ptr);

  template <int SYNC=__DEVICE_SYNC__>
  int host2device(char* host_ptr, char* dev_handle, uintptr_t dev_ptr, size_t len);

  template <int SYNC=__DEVICE_SYNC__>
  int device2host(char* dev_handle, uintptr_t dev_ptr, char* host_ptr, size_t len);

  void wait(int lock_idx);

}//end namespace
}//end namespace


#include <device_wrapper.txx>

#endif //_PVFMM_DEVICE_WRAPPER_HPP_
