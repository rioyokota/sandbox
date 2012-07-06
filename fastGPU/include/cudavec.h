#ifndef cudavec_h
#define cudavec_h
#include <cstdio>

#define CU_SAFE_CALL(err)  __checkCudaErrors (err, __FILE__, __LINE__)
inline void __checkCudaErrors(cudaError err, const char *file, const int line ) {
  if(cudaSuccess != err) {
    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
    exit(-1);
  }
}

template<typename T>
class cudaVec {
private:
  int SIZE;
  T *HOST;
  T *DEVC;

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

  void alloc(int n) {
    dealloc();
    SIZE = n;
    HOST = (T*)malloc(SIZE*sizeof(T));
    CU_SAFE_CALL(cudaMalloc((T**)&DEVC, SIZE*sizeof(T)));
  }

  void zeros() {
    CU_SAFE_CALL(cudaMemset((void*)DEVC, 0, SIZE*sizeof(T)));
  }

  void ones() {
    CU_SAFE_CALL(cudaMemset((void*)DEVC, 1, SIZE*sizeof(T)));
  }

  void d2h() {
    CU_SAFE_CALL(cudaMemcpy(HOST, DEVC, SIZE*sizeof(T), cudaMemcpyDeviceToHost));
  }

  void d2h(int n) {
    CU_SAFE_CALL(cudaMemcpy(HOST, DEVC, n*sizeof(T), cudaMemcpyDeviceToHost));
  }

  void h2d() {
    CU_SAFE_CALL(cudaMemcpy(DEVC, HOST, SIZE*sizeof(T), cudaMemcpyHostToDevice ));
  }

  void h2d(int n) {
    CU_SAFE_CALL(cudaMemcpy(DEVC, HOST, n*sizeof(T), cudaMemcpyHostToDevice));
  }

  void tex(const char *symbol) {
    const textureReference *texref;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    CU_SAFE_CALL(cudaGetTextureReference(&texref,symbol));
    CU_SAFE_CALL(cudaBindTexture(0, texref, (void*)DEVC, &channelDesc, SIZE*sizeof(T)));
  }

  T& operator[] (int i) const { return HOST[i]; }
  T* host() const { return HOST; }
  T* devc() const { return DEVC; }
  int size() const { return SIZE; }
};
#endif
