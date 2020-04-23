#include <cstdio>
#include <CL/cl.h>

const char *source = "__kernel void memcpy(__global float* a) {"\
  "__local float b[2];"\
  "barrier(CLK_LOCAL_MEM_FENCE);"\
  "b[get_local_id(0)] = 10 * get_group_id(0) + get_local_id(0);"\
  "barrier(CLK_LOCAL_MEM_FENCE);"\
  "a[get_global_id(0)] = b[get_local_id(0)];}";

int main() {
  cl_platform_id platform;
  cl_device_id device;
  clGetPlatformIDs(1, &platform, NULL);
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
  cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, NULL);
  cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);
  clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  cl_kernel kernel = clCreateKernel(program, "memcpy", NULL);

  int size = 4 * sizeof(float);
  cl_mem a = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, NULL, NULL);
  float *b = (float*) malloc(size);
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &a);
  size_t global = 4, local = 2;
  clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
  clEnqueueReadBuffer(commandQueue, a, CL_TRUE, 0, size, b, 0, NULL, NULL );
  for (int i=0; i<4; i++) printf("%f\n",b[i]);
  clReleaseMemObject(a);
  delete[] b;

  clFinish(commandQueue);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(commandQueue);
  clReleaseContext(context);
}
