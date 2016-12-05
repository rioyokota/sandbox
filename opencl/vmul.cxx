#include <iostream>
#include <CL/cl.h>

#define DATA_SIZE (1024*1240)

const char *KernelSource = "\n"      \
  "__kernel void square(                    \n" \
  "   __global float* input,                \n" \
  "   __global float* output,               \n" \
  "   const unsigned int count)             \n" \
  "{                                        \n" \
  "   int i = get_global_id(0);             \n" \
  "   if(i < count)                         \n" \
  "       output[i] = input[i] * input[i];  \n" \
  "}                                        \n" \
  "\n";

int main(int argc, char* argv[])
{
  int devType=CL_DEVICE_TYPE_ALL;

  cl_int err;
  size_t global;
  size_t local;
  cl_platform_id platform_id;
  cl_device_id device_id;
  cl_context context;
  cl_command_queue command_queue;
  cl_program program;
  cl_kernel kernel;

  // Connect to a compute device
  err = clGetPlatformIDs(1, &platform_id, NULL);
  if (err != CL_SUCCESS) {
    std::cerr << "Error: Failed to find a platform!" << std::endl;
    return EXIT_FAILURE;
  }

  // Get a device of the appropriate type
  err = clGetDeviceIDs(platform_id, devType, 1, &device_id, NULL);
  if (err != CL_SUCCESS) {
    std::cerr << "Error: Failed to create a device group!" << std::endl;
    return EXIT_FAILURE;
  }

  // Create a compute context
  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  if (!context) {
    std::cerr << "Error: Failed to create a compute context!" << std::endl;
    return EXIT_FAILURE;
  }

  // Create a command command_queue
  command_queue = clCreateCommandQueue(context, device_id, 0, &err);
  if (!command_queue) {
    std::cerr << "Error: Failed to create a command command_queue!" << std::endl;
    return EXIT_FAILURE;
  }

  // Create the compute program from the source buffer
  program = clCreateProgramWithSource(context, 1,
                                      (const char **) &KernelSource,
                                      NULL, &err);
  if (!program) {
    std::cerr << "Error: Failed to create compute program!" << std::endl;
    return EXIT_FAILURE;
  }

  // Build the program executable
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t len;
    char buffer[2048];

    std::cerr << "Error: Failed to build program executable!" << std::endl;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                          sizeof(buffer), buffer, &len);
    std::cerr << buffer << std::endl;
    exit(1);
  }

  // Create the compute kernel in the program
  kernel = clCreateKernel(program, "square", &err);
  if (!kernel || err != CL_SUCCESS) {
    std::cerr << "Error: Failed to create compute kernel!" << std::endl;
    exit(1);
  }

  // create data for the run
  float* data = new float[DATA_SIZE];    // original data set given to device
  float* results = new float[DATA_SIZE]; // results returned from device
  unsigned int correct;               // number of correct results returned
  cl_mem input;                       // device memory used for the input array
  cl_mem output;                      // device memory used for the output array

  // Fill the vector with random float values
  int count = DATA_SIZE;
  for(int i = 0; i < count; i++)
    data[i] = rand() / (float)RAND_MAX;

  // Create the device memory vectors
  //
  input = clCreateBuffer(context,  CL_MEM_READ_ONLY,
                         sizeof(float) * count, NULL, NULL);
  output = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                          sizeof(float) * count, NULL, NULL);
  if (!input || !output) {
    std::cerr << "Error: Failed to allocate device memory!" << std::endl;
    exit(1);
  }

  // Transfer the input vector into device memory
  err = clEnqueueWriteBuffer(command_queue, input,
                             CL_TRUE, 0, sizeof(float) * count,
                             data, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    std::cerr << "Error: Failed to write to source array!" << std::endl;
    exit(1);
  }

  // Set the arguments to the compute kernel
  err = 0;
  err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
  err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &count);
  if (err != CL_SUCCESS) {
    std::cerr << "Error: Failed to set kernel arguments! " << err << std::endl;
    exit(1);
  }

  // Get the maximum work group size for executing the kernel on the device
  err = clGetKernelWorkGroupInfo(kernel, device_id,
                                 CL_KERNEL_WORK_GROUP_SIZE,
                                 sizeof(local), &local, NULL);
  if (err != CL_SUCCESS) {
    std::cerr << "Error: Failed to retrieve kernel work group info! "
         <<  err << std::endl;
    exit(1);
  }

  // Execute the kernel over the vector using the
  // maximum number of work group items for this device
  global = count;
  err = clEnqueueNDRangeKernel(command_queue, kernel,
                               1, NULL, &global, &local,
                               0, NULL, NULL);
  if (err) {
    std::cerr << "Error: Failed to execute kernel!" << std::endl;
    return EXIT_FAILURE;
  }

  // Wait for all command_queue to complete
  clFinish(command_queue);

  // Read back the results from the device to verify the output
  //
  err = clEnqueueReadBuffer( command_queue, output,
                             CL_TRUE, 0, sizeof(float) * count,
                             results, 0, NULL, NULL );
  if (err != CL_SUCCESS) {
    std::cerr << "Error: Failed to read output array! " <<  err << std::endl;
    exit(1);
  }

  // Validate our results
  //
  correct = 0;
  for(int i = 0; i < count; i++) {
    if(results[i] == data[i] * data[i])
      correct++;
  }

  // Print a brief summary detailing the results
  std::cout << "Computed " << correct << "/" << count << " correct values" << std::endl;
  std::cout << "Computed " << 100.f * (float)correct/(float)count
       << "% correct values" << std::endl;

  // Shutdown and cleanup
  delete [] data; delete [] results;

  clReleaseMemObject(input);
  clReleaseMemObject(output);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);

  return 0;
}
