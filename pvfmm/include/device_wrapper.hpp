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

  void wait(int lock_idx);

}//end namespace
}//end namespace


#include <device_wrapper.txx>

#endif //_PVFMM_DEVICE_WRAPPER_HPP_
