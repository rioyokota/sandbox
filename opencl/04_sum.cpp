#include <cstdio>
#include <CL/cl.h>

const char *source = "__kernel void memcpy(__global int* a, __global int* sum) {"\
  "atomic_add(sum,a[get_global_id(0)]);}";

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

  int size = 4 * sizeof(int);
  cl_mem a = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, NULL);
  int *b = (int*) malloc(size);
  for (int i=0; i<4; i++) b[i] = 1;
  cl_mem sum = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);
  int *c = (int*) malloc(sizeof(int));
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &a);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &sum);
  size_t global = 4, local = 2;
  clEnqueueWriteBuffer(commandQueue, a, CL_TRUE, 0, size, b, 0, NULL, NULL);
  clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
  clEnqueueReadBuffer(commandQueue, sum, CL_TRUE, 0, sizeof(int), c, 0, NULL, NULL );
  printf("%d\n",*c);
  clReleaseMemObject(a);
  clReleaseMemObject(sum);
  delete[] b;
  delete[] c;

  clFinish(commandQueue);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(commandQueue);
  clReleaseContext(context);
}
