#include <cstdio>
#include <CL/cl.h>

const char *source = "__kernel void hello() {"\
  "printf(\"Hello GPU\\n\"); }";

int main() {
  cl_platform_id platform;
  cl_device_id device;
  clGetPlatformIDs(1, &platform, NULL);
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
  cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, NULL);
  cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);
  clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  cl_kernel kernel = clCreateKernel(program, "hello", NULL);
  printf("Hello CPU\n");
  size_t global = 1, local = 1;
  clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
  clFinish(commandQueue);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(commandQueue);
  clReleaseContext(context);
}
