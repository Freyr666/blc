#include "Alg.hpp"

#include <iostream>

#define RET_STATUS(st, rst) {\
    if (st)  \
      rst = st;}

const char * s_alg =
  "#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable \n"
  "#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics: enable \n"
  "#pragma OPENCL EXTENSION cl_khr_fp64 : enable \n"
  "void atomic_add_float (global float *ptr, float temp){ \n"
  " unsigned int newVal; \n"
  " unsigned int prevVal; \n"
  "  do{ \n"
  "          prevVal = as_uint(*ptr); \n"
  "          newVal = as_uint(temp + *ptr); \n"
  "  } while (atomic_cmpxchg((global unsigned int *)ptr, prevVal, newVal) != prevVal); \n"
  "}\n"
  "                                                "
  "__kernel void alg(\n"  //calcs hprof for each row
  "         __global const uchar* pic,\n"
  "         __global const uchar* picprev,\n"
  "         __global float* result,\n"
  "         int hsize, int vsize, uchar valbk, uchar valsm)\n"
  "{    \n"
  "  size_t globalId = get_global_id(0); \n"
  "  size_t groupSize = get_num_groups(0); \n"
  "  size_t localSize = get_local_size(0); \n"
  "  size_t groupId = get_group_id(0); \n"
  "  size_t localId = get_local_id(0); \n"
  "  size_t globalSize = get_global_size(0); \n"
  "  int black = 0;\n"
  "  int same = 0; \n"
  "  int dif = 0; \n"
  "  int col = 0; \n"
  "  float hacc = 0; \n"
  "  if (globalId >= hsize) return; \n"
  "  for (uint i = 0; i < vsize; i++){ \n"
  "    if (pic[globalId + i*hsize] < valbk) black++; \n"
  "    int diftmp = abs(pic[globalId + i*hsize] - picprev[globalId + i*hsize]); \n"
  "    if (diftmp < valsm) same++; \n"
  "    dif += diftmp; \n"
  "    col += pic[globalId + i*hsize]; \n"
  "    if ( globalId / 4 ){ \n"
  "      float sum, sub, subNext, subPrev; \n"
  "      sub = (float)abs(pic[globalId + i*hsize - 1] - pic[globalId + i*hsize]);\n"
  "      subNext = (float)abs(pic[globalId + i*hsize] - pic[globalId + i*hsize + 1]);\n" 
  "      subPrev = (float)abs(pic[globalId + i*hsize - 2] - pic[globalId + i*hsize - 1]);\n"
  "      sum = (subNext + subPrev);\n"
  "      if (sum == 0) sum = 1; \n"
  "      else sum = sum / 2; \n"
  "      hacc += sub / sum; \n"
  "    } \n"
  "  }\n"
  "  barrier(CLK_LOCAL_MEM_FENCE); \n"
  "  atomic_add_float(&result[0], black); \n"
  "  atomic_add_float(&result[1], same); \n"
  "  atomic_add_float(&result[2], dif); \n"
  "  atomic_add_float(&result[3], col); \n"
  "  if ( globalId / 8 ) atomic_add_float(&result[4], hacc); \n"
  "  else atomic_add_float(&result[5], hacc); \n"
  // "  result[0] = 1000000; \n"
  // "  result[1] = 1000000; \n"
  "}\n";

Alg::Alg(int cls, int rws, int thrds, int pltfrm, uint8_t vbk, uint8_t vsm){
  cols = cls;
  rows = rws;
  threads = thrds;
  platform_n = pltfrm;
  valbk = vbk;
  valsm = vsm;
  global_threads = cls;
  percentage = new float[5];
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
  status = clGetDeviceIDs(platform[platform_n], CL_DEVICE_TYPE_CPU, 0, NULL, &n_devices);
  //RET_STATUS(status, exit_status);
  device_list = new cl_device_id [n_devices];
  status = clGetDeviceIDs(platform[platform_n], CL_DEVICE_TYPE_CPU, n_devices, device_list, NULL);
  //RET_STATUS(status, exit_status);
  // Create one OpenCL context for each device in the platform
  context = clCreateContext(NULL, n_devices, device_list, NULL, NULL, &status);
  queue = clCreateCommandQueue(context, *device_list, 0, &status);
  // Create memory buffers on the device for each vector
  clm_pic = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, cls*rws*sizeof(cl_uchar), NULL, &status);
  clm_picprev = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, cls*rws*sizeof(cl_uchar), NULL, &status);
  clm_res = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 6*sizeof(cl_float), NULL, &status);
  status = clFinish(queue);
  // Create programs from the kernel source
  prog = clCreateProgramWithSource(context, 1, (const char**)&s_alg, NULL, &status);
  status = clBuildProgram(prog, 1, device_list, NULL, NULL, NULL);
  // Create the OpenCL kernels
  kern = clCreateKernel(prog, "alg", &status);
  status = clSetKernelArg(kern, 0, sizeof(cl_mem), (void*)&clm_pic);
  status = clSetKernelArg(kern, 1, sizeof(cl_mem), (void*)&clm_picprev);
  status = clSetKernelArg(kern, 2, sizeof(cl_mem), (void*)&clm_res);
  status = clSetKernelArg(kern, 3, sizeof(cl_int), (void*)&cols);
  status = clSetKernelArg(kern, 4, sizeof(cl_int), (void*)&rows);
  status = clSetKernelArg(kern, 5, sizeof(cl_uchar), (void*)&valbk);
  status = clSetKernelArg(kern, 6, sizeof(cl_uchar), (void*)&valsm);
}

Alg::~Alg(){
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
Alg::eval(uint8_t *t, uint8_t *tp){
  float* result = (float*)clEnqueueMapBuffer(queue, clm_res, CL_FALSE, CL_MAP_READ, 0, 6*sizeof(cl_float), 0, NULL, NULL, &status);
  for (int i = 0; i < 6; i++) {
    result[i] = 0;
  }
  // Copy frame to the device
  status = clEnqueueWriteBuffer(queue, clm_pic, CL_FALSE, 0, cols * rows * sizeof(cl_uchar), t, 0, NULL, NULL);
  status = clEnqueueWriteBuffer(queue, clm_picprev, CL_FALSE, 0, cols * rows * sizeof(cl_uchar), tp, 0, NULL, NULL);
  status = clEnqueueWriteBuffer(queue, clm_res, CL_FALSE, 0, 6 * sizeof(float), result, 0, NULL, NULL);
  // Execute the OpenCL kernel on the list
  status = clEnqueueNDRangeKernel(queue, kern, 1, NULL, &global_threads, NULL, 0, NULL, NULL);
  status = clEnqueueReadBuffer(queue, clm_res, CL_FALSE, 0, 6*sizeof(float), result, 0, NULL, NULL);
  // Clean up and wait for all the comands to complete.
  status = clFlush(queue);
  status = clFinish(queue);
  for (int i = 0; i < 2; i++) {
    percentage[i] = result[i]*100.0 / (cols*rows);
  }
  for (int i = 0; i < 2; i++) {
    percentage[i + 2] = result[i + 2] / (cols*rows);
  }
  if (result[5] > 0.5)
    percentage[4] = result[4] / result[5];
  else
    percentage[4] = result[4] / 4;
  clEnqueueUnmapMemObject(queue, clm_res, result, 0, NULL, NULL);
  return percentage;
}

void*
Alg::eval(std::vector<uint8_t>* pic, std::vector<uint8_t>* picprev){
  uint8_t* t = reinterpret_cast<uint8_t*>(pic->data());
  uint8_t* tp = reinterpret_cast<uint8_t*>(picprev->data());
  // Copy frame to the device
  float* result = (float*)clEnqueueMapBuffer(queue, clm_res, CL_FALSE, CL_MAP_READ, 0, 6*sizeof(cl_float), 0, NULL, NULL, &status);
  for (int i = 0; i < 6; i++) {
    result[i] = 0;
  }
  // Copy frame to the device
  status = clEnqueueWriteBuffer(queue, clm_pic, CL_FALSE, 0, cols * rows * sizeof(cl_uchar), t, 0, NULL, NULL);
  status = clEnqueueWriteBuffer(queue, clm_picprev, CL_FALSE, 0, cols * rows * sizeof(cl_uchar), tp, 0, NULL, NULL);
  status = clEnqueueWriteBuffer(queue, clm_res, CL_FALSE, 0, 6 * sizeof(float), result, 0, NULL, NULL);
  // Execute the OpenCL kernel on the list
  status = clEnqueueNDRangeKernel(queue, kern, 1, NULL, &global_threads, NULL, 0, NULL, NULL);
  status = clEnqueueReadBuffer(queue, clm_res, CL_FALSE, 0, 6*sizeof(float), result, 0, NULL, NULL);
  // Clean up and wait for all the comands to complete.
  status = clFlush(queue);
  status = clFinish(queue);
  for (int i = 0; i < 2; i++) {
    percentage[i] = result[i]*100.0 / (cols*rows);
  }
  for (int i = 0; i < 2; i++) {
    percentage[i + 2] = result[i + 2] / (cols*rows);
  }
  if (result[5])
    percentage[4] = result[4] / result[5];
  else
    percentage[4] = result[4] / 4;
  clEnqueueUnmapMemObject(queue, clm_res, result, 0, NULL, NULL);
  return percentage;
}

char*
Alg::compilerlog(void){
  cl_int logStatus;
  char * buildLog = NULL;	
  size_t buildLogSize = 0;
  logStatus = clGetProgramBuildInfo(prog, *device_list, CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, &buildLogSize);
  buildLog = new char [buildLogSize];
  logStatus = clGetProgramBuildInfo(prog, *device_list, CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL);
  return buildLog;
}
