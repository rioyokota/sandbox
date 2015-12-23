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

  // MIC functions

  #define ALLOC alloc_if(1) free_if(0)
  #define FREE alloc_if(0) free_if(1)
  #define REUSE alloc_if(0) free_if(0)

  inline uintptr_t alloc_device_mic(char* dev_handle, size_t len){
    assert(dev_handle!=NULL);
    uintptr_t dev_ptr=(uintptr_t)NULL;
    #ifdef __INTEL_OFFLOAD
    #pragma offload target(mic:0) nocopy( dev_handle: length(len) ALLOC) out(dev_ptr)
    #else
    UNUSED(len);
    #endif
    {dev_ptr=(uintptr_t)dev_handle;}
    return dev_ptr;
  }

  inline void free_device_mic(char* dev_handle, uintptr_t dev_ptr){
    #ifdef __INTEL_OFFLOAD
    #pragma offload          target(mic:0) in( dev_handle: length(0) FREE)
    {
      assert(dev_ptr==(uintptr_t)dev_handle);
    }
    #else
    UNUSED(dev_handle);
    UNUSED(dev_ptr);
    #endif
  }

  inline int host2device_mic(char* host_ptr, char* dev_handle, uintptr_t dev_ptr, size_t len){
    #ifdef __INTEL_OFFLOAD
    int wait_lock_idx=MIC_Lock::curr_lock();
    int lock_idx=MIC_Lock::get_lock();
    if(dev_handle==host_ptr){
      #pragma offload target(mic:0)  in( dev_handle        :              length(len)  REUSE ) signal(&MIC_Lock::lock_vec[lock_idx])
      {
        assert(dev_ptr==(uintptr_t)dev_handle);
        MIC_Lock::wait_lock(wait_lock_idx);
        MIC_Lock::release_lock(lock_idx);
      }
    }else{
      #pragma offload target(mic:0)  in(host_ptr   [0:len] : into ( dev_handle[0:len]) REUSE ) signal(&MIC_Lock::lock_vec[lock_idx])
      {
        assert(dev_ptr==(uintptr_t)dev_handle);
        MIC_Lock::wait_lock(wait_lock_idx);
        MIC_Lock::release_lock(lock_idx);
      }
    }
    return lock_idx;
    #else
    UNUSED(host_ptr);
    UNUSED(dev_handle);
    UNUSED(dev_ptr);
    UNUSED(len);
    #endif
    return -1;
  }

  inline int device2host_mic(char* dev_handle, uintptr_t dev_ptr, char* host_ptr, size_t len){
    #ifdef __INTEL_OFFLOAD
    int wait_lock_idx=MIC_Lock::curr_lock();
    int lock_idx=MIC_Lock::get_lock();
    if(dev_handle==host_ptr){
      #pragma offload target(mic:0) out( dev_handle        :              length(len)  REUSE ) signal(&MIC_Lock::lock_vec[lock_idx])
      {
        assert(dev_ptr==(uintptr_t)dev_handle);
        MIC_Lock::wait_lock(wait_lock_idx);
        MIC_Lock::release_lock(lock_idx);
      }
    }else{
      #pragma offload target(mic:0) out( dev_handle[0:len] : into (host_ptr   [0:len]) REUSE ) signal(&MIC_Lock::lock_vec[lock_idx])
      {
        assert(dev_ptr==(uintptr_t)dev_handle);
        MIC_Lock::wait_lock(wait_lock_idx);
        MIC_Lock::release_lock(lock_idx);
      }
    }
    return lock_idx;
    #else
    UNUSED(host_ptr);
    UNUSED(dev_handle);
    UNUSED(dev_ptr);
    UNUSED(len);
    #endif
    return -1;
  }

  inline void wait_mic(int lock_idx){
    #ifdef __INTEL_OFFLOAD
    MIC_Lock::wait_lock(lock_idx);
    #else
    UNUSED(lock_idx);
    #endif
  }



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

}


  // Implementation of MIC_Lock

  #ifdef __MIC__
  #define have_mic 1
  #else
  #define have_mic 0
  #endif

  #define NUM_LOCKS 1000000
  inline void MIC_Lock::init(){
    #ifdef __INTEL_OFFLOAD
    if(have_mic) abort();// Cannot be called from MIC.

    lock_idx=0;
    lock_vec.Resize(NUM_LOCKS);
    lock_vec.SetZero();
    lock_vec_=lock_vec.AllocDevice(false);
    {for(size_t i=0;i<NUM_LOCKS;i++) lock_vec [i]=1;}
    #pragma offload target(mic:0)
    {for(size_t i=0;i<NUM_LOCKS;i++) lock_vec_[i]=1;}
    #endif
  }

  inline int MIC_Lock::get_lock(){
    #ifdef __INTEL_OFFLOAD
    if(have_mic) abort();// Cannot be called from MIC.

    int idx;
    #pragma omp critical
    {
      if(lock_idx==NUM_LOCKS-1){
        int wait_lock_idx=-1;
        wait_lock_idx=MIC_Lock::curr_lock();
        MIC_Lock::wait_lock(wait_lock_idx);
        #pragma offload target(mic:0)
        {MIC_Lock::wait_lock(wait_lock_idx);}
        MIC_Lock::init();
      }
      idx=lock_idx;
      lock_idx++;
      assert(lock_idx<NUM_LOCKS);
    }
    return idx;
    #else
    return -1;
    #endif
  }
  #undef NUM_LOCKS

  inline int MIC_Lock::curr_lock(){
    #ifdef __INTEL_OFFLOAD
    if(have_mic) abort();// Cannot be called from MIC.
    return lock_idx-1;
    #else
    return -1;
    #endif
  }

  inline void MIC_Lock::release_lock(int idx){ // Only call from inside an offload section
    #if defined(__INTEL_OFFLOAD) && defined(__MIC__)
    if(idx>=0) lock_vec_[idx]=0;
    #else
    UNUSED(idx);
    #endif
  }

  inline void MIC_Lock::wait_lock(int idx){
    #ifdef __INTEL_OFFLOAD
    #ifdef __MIC__
    if(idx>=0) while(lock_vec_[idx]==1){
      _mm_delay_32(8192);
    }
    #else
    if(idx<0 || lock_vec[idx]==0) return;
    if(lock_vec[idx]==2){
      while(lock_vec[idx]==2);
      return;
    }
    lock_vec[idx]=2;
    #pragma offload_wait target(mic:0) wait(&lock_vec[idx])
    lock_vec[idx]=0;
    #endif
    #else
    UNUSED(idx);
    #endif
  }

}//end namespace
