/**
 * \file pvfmm_common.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 12-10-2010
 * \brief This file contains common definitions.
 */

#include "pvfmm_config.h"

#ifndef _PVFMM_COMMON_HPP_
#define _PVFMM_COMMON_HPP_

//Define NULL
#ifndef NULL
#define NULL 0
#endif

//Disable assert checks.
#ifndef NDEBUG
#define NDEBUG
#endif

//Enable profiling
#define __PROFILE__ 5

//Verbose
//#define __VERBOSE__

#define MAX_DEPTH 15

#define BC_LEVELS 60

#define RAD0 1.05 //Radius of upward equivalent (downward check) surface.
#define RAD1 2.95 //Radius of downward equivalent (upward check) surface.

#define COORD_DIM 3
#define COLLEAGUE_COUNT 27 // 3^COORD_DIM

#define MEM_ALIGN 64
#define DEVICE_BUFFER_SIZE 1024LL //in MB
#define V_BLK_CACHE 25 //in KB
#define GLOBAL_MEM_BUFF 1024LL*0LL //in MB

#define UNUSED(x) (void)(x) // to ignore unused variable warning.

#include <math_utils.hpp>

#endif //_PVFMM_COMMON_HPP_
