#include <cstdio>
#include <CL/cl.h>

int main() {
  cl_platform_id *platformId = new cl_platform_id [3];
  cl_uint platformCount;
  clGetPlatformIDs(3, platformId, &platformCount);
  char *name = new char [32];
  char *version = new char [32];
  for (int i=0; i<platformCount; i++) {
    clGetPlatformInfo(platformId[i], CL_PLATFORM_NAME, 32, name, NULL);
    clGetPlatformInfo(platformId[i], CL_PLATFORM_VERSION, 32, version, NULL);
    printf("Platform %d: %s, %s\n", i, name, version);
  }
  delete[] name;
  delete[] version;
  delete[] platformId;
}
