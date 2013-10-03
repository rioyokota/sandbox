#pragma once

template<typename T>
class cudaVec {
private:
  int SIZE;
  T * HOST;
  T * DEVC;

  void dealloc() {
    if( SIZE != 0 ) {
      SIZE = 0;
      free(HOST);
      cudaFree(DEVC);
    }
  }

public:
  cudaVec() : SIZE(0), HOST(NULL), DEVC(NULL) {}
  ~cudaVec() {
    dealloc();
  }

  void alloc(int size) {
    dealloc();
    SIZE = size;
    HOST = (T*)malloc(SIZE*sizeof(T));
    CUDA_SAFE_CALL(cudaMalloc((T**)&DEVC, SIZE*sizeof(T)));
  }

  void zeros() {
    CUDA_SAFE_CALL(cudaMemset(DEVC, 0, SIZE*sizeof(T)));
  }

  void ones() {
    CUDA_SAFE_CALL(cudaMemset(DEVC, 1, SIZE*sizeof(T)));
  }

  void d2h() {
    CUDA_SAFE_CALL(cudaMemcpy(HOST, DEVC, SIZE*sizeof(T), cudaMemcpyDeviceToHost));
  }

  void d2h(int size) {
    CUDA_SAFE_CALL(cudaMemcpy(HOST, DEVC, size*sizeof(T), cudaMemcpyDeviceToHost));
  }

  void h2d() {
    CUDA_SAFE_CALL(cudaMemcpy(DEVC, HOST, SIZE*sizeof(T), cudaMemcpyHostToDevice ));
  }

  void h2d(int size) {
    CUDA_SAFE_CALL(cudaMemcpy(DEVC, HOST, size*sizeof(T), cudaMemcpyHostToDevice));
  }

  void bindTexture(texture<T,1,cudaReadModeElementType> &tex) {
    tex.addressMode[0] = cudaAddressModeWrap;
    tex.addressMode[1] = cudaAddressModeWrap;
    tex.filterMode     = cudaFilterModePoint;
    tex.normalized     = false;
    CUDA_SAFE_CALL(cudaBindTexture(0, tex, DEVC, SIZE*sizeof(T)));
  }

  void unbindTexture(texture<T,1,cudaReadModeElementType> &tex) {
    CUDA_SAFE_CALL(cudaUnbindTexture(tex));
  }

  T& operator[] (int i) const { return HOST[i]; }
  T* host() const { return HOST; }
  T* devc() const { return DEVC; }
  int size() const { return SIZE; }
};
