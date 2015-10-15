#include "IsBlack.hpp"

#include <iostream>

#define RET_STATUS(st, rst) {\
    if (st)  \
      rst = st;}

const char * s_isblck =
  "#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable \n"
  "#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics: enable \n"
  "#pragma OPENCL EXTENSION cl_khr_fp64 : enable \n"
  "                                                "
  "__kernel void isblck(\n"  //calcs hprof for each row
  "         __global const uchar* pic,\n"
  "         __global const uchar* picprev,\n"
  "         __global int* result,\n"
  "         int hsize, int vsize, uchar val)\n"
  "{    \n"
  "  size_t globalId = get_global_id(0); \n"
  "  size_t groupSize = get_num_groups(0); \n"
  "  size_t localSize = get_local_size(0); \n"
  "  size_t groupId = get_group_id(0); \n"
  "  size_t localId = get_local_id(0); \n"
  "  size_t globalSize = get_global_size(0); \n"
  "  int black = 0;\n"
  "  int same = 0; \n"
  "  if (globalId >= hsize) return; \n"
  "  for (uint i = 0; i < vsize; i++){ \n"
  "      if (pic[globalId + i*hsize] < val) black++; \n"
  "      if (pic[globalId + i*hsize] == picprev[globalId + i*hsize]) same++; \n"
  "  }\n"
  "  barrier(CLK_LOCAL_MEM_FENCE); \n"
  "  atomic_add(&result[0], black); \n"
  "  atomic_add(&result[1], same); \n"
  // "  result[0] = 1000000; \n"
  // "  result[1] = 1000000; \n"
  "}\n";

IsBlack::IsBlack(int cls, int rws, int thrds, int pltfrm, uint8_t v){
  cols = cls;
  rows = rws;
  threads = thrds;
  platform_n = pltfrm;
  val = v;
  global_threads = cls;
  percentage = new float[2];
  //opencl init
  platform = NULL;
  device_list = NULL;
  //Set up the Platform
  status = clGetPlatformIDs(0, NULL, &n_platforms);
  //RET_STATUS(status, exit_status);
  platform = new cl_platform_id[n_platforms];
  status = clGetPlatformIDs(n_platforms, platform, NULL);
  //RET_STATUS(status, exit_status);
  //Get the devices list and choose the device you want to run on
  status = clGetDeviceIDs(platform[platform_n], CL_DEVICE_TYPE_GPU, 0, NULL, &n_devices);
  //RET_STATUS(status, exit_status);
  device_list = new cl_device_id [n_devices];
  status = clGetDeviceIDs(platform[platform_n], CL_DEVICE_TYPE_GPU, n_devices, device_list, NULL);
  //RET_STATUS(status, exit_status);
  // Create one OpenCL context for each device in the platform
  context = clCreateContext(NULL, n_devices, device_list, NULL, NULL, &status);
  queue = clCreateCommandQueue(context, *device_list, 0, &status);
  // Create memory buffers on the device for each vector
  clm_pic = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, cls*rws*sizeof(cl_uchar), NULL, &status);
  clm_picprev = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, cls*rws*sizeof(cl_uchar), NULL, &status);
  clm_res = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 2*sizeof(cl_int), NULL, &status);
  status = clFinish(queue);
  // Create programs from the kernel source
  prog = clCreateProgramWithSource(context, 1, (const char**)&s_isblck, NULL, &status);
  status = clBuildProgram(prog, 1, device_list, NULL, NULL, NULL);
  // Create the OpenCL kernels
  kern = clCreateKernel(prog, "isblck", &status);
  status = clSetKernelArg(kern, 0, sizeof(cl_mem), (void*)&clm_pic);
  status = clSetKernelArg(kern, 1, sizeof(cl_mem), (void*)&clm_picprev);
  status = clSetKernelArg(kern, 2, sizeof(cl_mem), (void*)&clm_res);
  status = clSetKernelArg(kern, 3, sizeof(cl_int), (void*)&cols);
  status = clSetKernelArg(kern, 4, sizeof(cl_int), (void*)&rows);
  status = clSetKernelArg(kern, 5, sizeof(cl_uchar), (void*)&val);
}

IsBlack::~IsBlack(){
  status = clReleaseKernel(kern);
  status = clReleaseProgram(prog);
  status = clReleaseMemObject(clm_pic);
  status = clReleaseMemObject(clm_picprev);
  status = clReleaseMemObject(clm_res);
  status = clReleaseCommandQueue(queue);
  status = clReleaseContext(context);
  delete percentage;
  delete platform;
  delete device_list;
}

void*
IsBlack::eval(uint8_t *t, uint8_t *tp){
  int* result = (int*)clEnqueueMapBuffer(queue, clm_res, CL_FALSE, CL_MAP_READ, 0, 2*sizeof(cl_int), 0, NULL, NULL, &status);
  for (int i = 0; i < 2; i++) {
    result[i] = 0;
  }
  // Copy frame to the device
  status = clEnqueueWriteBuffer(queue, clm_pic, CL_FALSE, 0, cols * rows * sizeof(cl_uchar), t, 0, NULL, NULL);
  status = clEnqueueWriteBuffer(queue, clm_picprev, CL_FALSE, 0, cols * rows * sizeof(cl_uchar), tp, 0, NULL, NULL);
  status = clEnqueueWriteBuffer(queue, clm_res, CL_FALSE, 0, 2 * sizeof(int), result, 0, NULL, NULL);
  // Execute the OpenCL kernel on the list
  status = clEnqueueNDRangeKernel(queue, kern, 1, NULL, &global_threads, NULL, 0, NULL, NULL);
  status = clEnqueueReadBuffer(queue, clm_res, CL_FALSE, 0, 2*sizeof(int), result, 0, NULL, NULL);
  // Clean up and wait for all the comands to complete.
  status = clFlush(queue);
  status = clFinish(queue);
  for (int i = 0; i < 2; i++) {
    percentage[i] = result[i]*100.0 / (cols*rows);
  }
  clEnqueueUnmapMemObject(queue, clm_res, result, 0, NULL, NULL);
  return percentage;
}

void*
IsBlack::eval(std::vector<uint8_t>* pic, std::vector<uint8_t>* picprev){
  uint8_t* t = reinterpret_cast<uint8_t*>(pic->data());
  uint8_t* tp = reinterpret_cast<uint8_t*>(picprev->data());
  // Copy frame to the device
  int* result = (int*)clEnqueueMapBuffer(queue, clm_res, CL_FALSE, CL_MAP_READ, 0, 2*sizeof(cl_int), 0, NULL, NULL, &status);
  for (int i = 0; i < 2; i++) {
    result[i] = 0;
  }
  // Copy frame to the device
  status = clEnqueueWriteBuffer(queue, clm_pic, CL_FALSE, 0, cols * rows * sizeof(cl_uchar), t, 0, NULL, NULL);
  status = clEnqueueWriteBuffer(queue, clm_picprev, CL_FALSE, 0, cols * rows * sizeof(cl_uchar), tp, 0, NULL, NULL);
  status = clEnqueueWriteBuffer(queue, clm_res, CL_FALSE, 0, 2 * sizeof(int), result, 0, NULL, NULL);
  // Execute the OpenCL kernel on the list
  status = clEnqueueNDRangeKernel(queue, kern, 1, NULL, &global_threads, NULL, 0, NULL, NULL);
  status = clEnqueueReadBuffer(queue, clm_res, CL_FALSE, 0, 2*sizeof(int), result, 0, NULL, NULL);
  // Clean up and wait for all the comands to complete.
  status = clFlush(queue);
  status = clFinish(queue);
  for (int i = 0; i < 2; i++) {
    percentage[i] = result[i]*100.0 / (cols*rows);
  }
  clEnqueueUnmapMemObject(queue, clm_res, result, 0, NULL, NULL);
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
