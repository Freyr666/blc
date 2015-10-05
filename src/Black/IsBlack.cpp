#include "IsBlack.hpp"

const char * s_isblck =
  //"#pragma OPENCL EXTENSION cl_khr_fp64 : enable  \n"
  "__kernel void pic2hprof(\n"  //calcs hprof for each row
  "         __global const uchar* pic,\n"
  "         __global int* result,\n"
  "         __local int* part, \n"
  "         int hsize, int vsize, uchar val)\n"
  "{    \n"
  "  size_t globalId = get_global_id(0); \n"
  "  size_t groupSize = get_num_groups(0); \n"
  "  size_t localSize = get_local_size(0); \n"
  "  size_t groupId = get_group_id(0); \n"
  "  size_t localId = get_local_id(0); \n"
  "  size_t globalSize = get_global_size(0); \n"
  "  if (pic[globalId] < val) part[localId] = 1; \n"
  "  if (globalId >= hsize) return;"
  "  for (uint i = 0; i > 1; i >>= 1) { \n"
  "      barrier(CLK_LOCAL_MEM_FENCE); \n"
  "      if (localId < i) \n"
  "         part[localId] += part[localId + i];\n"
  "  }\n"
  "  if (localId == 0) \n"
  "      result[groupId] = part[0];\n"
  "}\n";

IsBlack::IsBlack(int cls, int rws, int thrds, int pltfrm, uint8_t v){
  cols = cls;
  rows = rws;
  threads = thrds;
  platform_n = pltfrm;
  val = v;
  global_threads = rws;
  result = new int[rws*cls/threads];
  percentage = new float;
  *percentage = 0;
  //opencl init
  platform = NULL;
  device_list = NULL;
  //Set up the Platform
  status = clGetPlatformIDs(0, NULL, &n_platforms);
  platform = new cl_platform_id[n_platforms];
  status = clGetPlatformIDs(n_platforms, platform, NULL);
  //Get the devices list and choose the device you want to run on
  status = clGetDeviceIDs(platform[platform_n], CL_DEVICE_TYPE_GPU, 0, NULL, &n_devices);
  device_list = new cl_device_id [n_devices];
  status = clGetDeviceIDs(platform[platform_n], CL_DEVICE_TYPE_GPU, n_devices, device_list, NULL);
  // Create one OpenCL context for each device in the platform
  context = clCreateContext(NULL, n_devices, device_list, NULL, NULL, &status);
  queue = clCreateCommandQueue(context, *device_list, 0, &status);
  // Create memory buffers on the device for each vector
  clm_pic = clCreateBuffer(context, CL_MEM_READ_ONLY, cls*rws*sizeof(cl_uchar), NULL, &status);
  clm_res = clCreateBuffer(context, CL_MEM_READ_WRITE, rws*sizeof(cl_int), NULL, &status);
  status = clFinish(queue);
  // Create programs from the kernel source
  prog = clCreateProgramWithSource(context, 1, (const char**)&s_isblck, NULL, &status);
  status = clBuildProgram(prog, 1, device_list, NULL, NULL, NULL);
  // Create the OpenCL kernels
  kern = clCreateKernel(prog, "pic2hprof", &status);
  status = clSetKernelArg(kern, 1, sizeof(cl_mem), (void*)&clm_res);
  status = clSetKernelArg(kern, 2, threads*sizeof(int), NULL);
  status = clSetKernelArg(kern, 3, sizeof(cl_int), (void*)&cols);
  status = clSetKernelArg(kern, 4, sizeof(cl_int), (void*)&rows);
  status = clSetKernelArg(kern, 5, sizeof(cl_uchar), (void*)&val);
}

IsBlack::~IsBlack(){
  status = clReleaseKernel(kern);
  status = clReleaseProgram(prog);
  status = clReleaseMemObject(clm_pic);
  status = clReleaseMemObject(clm_res);
  status = clReleaseCommandQueue(queue);
  status = clReleaseContext(context);
  delete percentage;
  delete platform;
  delete device_list;
}

void*
IsBlack::eval(uint8_t *t){
  // Copy frame to the device
  status = clEnqueueWriteBuffer(queue, clm_pic, CL_TRUE, 0, cols * rows * sizeof(cl_uchar), t, 0, NULL, NULL);
  //setting kernels args
  status = clSetKernelArg(kern, 0, sizeof(cl_mem), (void*)&clm_pic);
  // Execute the OpenCL kernel on the list
  status = clEnqueueNDRangeKernel(queue, kern, 1, NULL, &global_threads, &threads, 0, NULL, NULL);
  status = clEnqueueReadBuffer(queue, clm_res, CL_TRUE, 0, (rows*cols/threads)*sizeof(int), result, 0, NULL, NULL);
  // Clean up and wait for all the comands to complete.
  status = clFlush(queue);
  status = clFinish(queue);
    for (int i = 0; i < (rows*cols/threads); i++) {
    *percentage += result[i];
  }
  *percentage = (*percentage)*100.0 / (cols*rows);
  return percentage;
}

void*
IsBlack::eval(std::vector<uint8_t>* t){
  uint8_t* pic = reinterpret_cast<uint8_t*>(t->data());
  // Copy frame to the device
  status = clEnqueueWriteBuffer(queue, clm_pic, CL_TRUE, 0, cols * rows * sizeof(cl_uchar), pic, 0, NULL, NULL);
  // setting kernels args
  status = clSetKernelArg(kern, 0, sizeof(cl_mem), (void*)&clm_pic);
  // Execute the OpenCL kernel on the list
  status = clEnqueueNDRangeKernel(queue, kern, 1, NULL, &global_threads, &threads, 0, NULL, NULL);
  status = clEnqueueReadBuffer(queue, clm_res, CL_TRUE, 0, (rows*cols/threads)*sizeof(int), result, 0, NULL, NULL);
  // Clean up and wait for all the comands to complete.
  status = clFlush(queue);
  status = clFinish(queue);
  for (int i = 0; i < (rows*cols/threads); i++) {
    *percentage += result[i];
  }
  *percentage = (*percentage)*100.0 / (cols*rows);
  return percentage;
}

char*
IsBlack::compilerlog(void){
  cl_int logStatus;							
  char * buildLog = NULL;						
  size_t buildLogSize = 0;
  logStatus = clGetProgramBuildInfo(prog, *device_list, CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, &buildLogSize);
  buildLog = new char [buildLogSize];
  logStatus = clGetProgramBuildInfo(prog, *device_list, CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL);
  return buildLog;
}
