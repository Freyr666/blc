#include "Optzd.hpp"

#include <iostream>

#define VEC_SIZE 4

  // "  barrier(CLK_LOCAL_MEM_FENCE);\n"

const char * s_pic2bsOpt =
  "#define VEC_SIZE 4\n"
  "__kernel __attribute__((vec_type_hint(float8)))\n"
  "  void pic2hprof(\n"  //calcs hprof for each row
  "         __global const uchar* pic,\n"
  "         __global float* sh,\n"
  "         int hsize, int vsize,\n"
  "         __local uchar4* col1,\n"
  "         __local uchar4* col2,\n"
  "         __local uchar4* col3,\n"
  "         __local uchar4* col4)\n"
  "{    \n"
  "  size_t globalId = get_global_id(0); \n"
  "  size_t groupSize = get_num_groups(0); \n"
  "  size_t localSize = get_local_size(0); \n"
  "  size_t groupId = get_group_id(0); \n"
  "  size_t localId = get_local_id(0); \n"
  "  size_t globalSize = get_global_size(0); \n"
  "  if (globalId >= hsize / 4) return;\n"
  "  for (int i = 0; i < vsize/VEC_SIZE; i++) { \n"
  "    int index = 4*(globalId); \n"
  "    col1[i] = (uchar4)(pic[index + (i*VEC_SIZE)*hsize - 2], \n"
  "                       pic[index + (i*VEC_SIZE + 1)*hsize - 2], \n"
  "                       pic[index + (i*VEC_SIZE + 2)*hsize - 2], \n"
  "                       pic[index + (i*VEC_SIZE + 3)*hsize - 2]); \n"
  "    col2[i] = (uchar4)(pic[index + (i*VEC_SIZE)*hsize - 1], \n"
  "                       pic[index + (i*VEC_SIZE + 1)*hsize - 1], \n"
  "                       pic[index + (i*VEC_SIZE + 2)*hsize - 1], \n"
  "                       pic[index + (i*VEC_SIZE + 3)*hsize - 1]); \n"
  "    col3[i] = (uchar4)(pic[index + (i*VEC_SIZE)*hsize], \n"
  "                       pic[index + (i*VEC_SIZE + 1)*hsize], \n"
  "                       pic[index + (i*VEC_SIZE + 2)*hsize], \n"
  "                       pic[index + (i*VEC_SIZE + 3)*hsize]); \n"
  "    col4[i] = (uchar4)(pic[index + (i*VEC_SIZE)*hsize + 1], \n"
  "                       pic[index + (i*VEC_SIZE + 1)*hsize + 1], \n"
  "                       pic[index + (i*VEC_SIZE + 2)*hsize + 1], \n"
  "                       pic[index + (i*VEC_SIZE + 3)*hsize + 1]); \n"
  "  }\n"
  "  barrier(CLK_LOCAL_MEM_FENCE);\n"
  "  float4 sum, sub, subNext, subPrev, hacc = (float4)(0.0f); \n"
  "  for (int i = 0; i < vsize/VEC_SIZE; i++) { \n"
  "         sub = convert_float4(col2[i] - col3[i]);\n"
  "         subNext = convert_float4(col3[i] - col4[i]);\n" 
  "         subPrev = convert_float4(col1[i] - col2[i]);\n"
  "         sum = (subNext + subPrev);\n"
  "         if (sum.x == 0) sum.x = 2; \n"
  "         if (sum.y == 0) sum.y = 2; \n"
  "         if (sum.z == 0) sum.z = 2; \n"
  "         if (sum.w == 0) sum.w = 2; \n"
  "         sum = sum / (float4)(2.0f); \n"
  "         hacc += sub / sum;} \n"
  "  sh[globalId] += dot(hacc, (float4)(1));\n"
  "}\n";

const char * s_pic2bs =
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
  "__kernel void pic2hprof(\n"  //calcs hprof for each row
  "         __global const uchar* pic,\n"
  "         __global float* sh,\n"
  "         int hsize, int vsize)\n"
  "{    \n"
  "  size_t globalId = get_global_id(0); \n"
  "  size_t groupSize = get_num_groups(0); \n"
  "  size_t localSize = get_local_size(0); \n"
  "  size_t groupId = get_group_id(0); \n"
  "  size_t localId = get_local_id(0); \n"
  "  size_t globalSize = get_global_size(0); \n"
  "  if (globalId >= hsize / 4) return;\n"
  "  float sum, sub, subNext, subPrev, hacc = 0; \n"
  "  for (int i = 0; i < vsize; i++) { \n"
  "         sub = (float)abs(pic[globalId*4 + i*hsize - 1] - pic[globalId*4 + i*hsize]);\n"
  "         subNext = (float)abs(pic[globalId*4 + i*hsize] - pic[globalId*4 + i*hsize + 1]);\n" 
  "         subPrev = (float)abs(pic[globalId*4 + i*hsize - 2] - pic[globalId*4 + i*hsize - 1]);\n"
  "         sum = (subNext + subPrev);\n"
  "         if (sum == 0) sum = 1; \n"
  "         else sum = sum / 2; \n"
  "         hacc += sub / sum;} \n"
  "  if (globalId % 2)\n"
  "     atomic_add_float(&sh[0], hacc);\n"
  "  else \n"
  "     atomic_add_float(&sh[1], hacc);\n"
  "}\n";

Optzd::Optzd(int cls, int rws, int thrds, int pltfrm){
  //vals init
  cols = cls;
  rows = rws;
  threads = thrds;
  global_threads = cls/4;
  BS = new double;
  // sh = new float[global_threads];
  //opencl initj*localSize + localId
  platform = NULL;
  device_list = NULL;
  //Set up the Platform
  status = clGetPlatformIDs(0, NULL, &n_platforms);
  platform = new cl_platform_id[n_platforms];
  status = clGetPlatformIDs(n_platforms, platform, NULL);
  //Get the devices list and choose the device you want to run on
  status = clGetDeviceIDs(platform[pltfrm], CL_DEVICE_TYPE_GPU, 0, NULL, &n_devices);
  device_list = new cl_device_id [n_devices];
  status = clGetDeviceIDs(platform[pltfrm], CL_DEVICE_TYPE_GPU, n_devices, device_list, NULL);
  // Create one OpenCL context for each device in the platform
  context = clCreateContext(NULL, n_devices, device_list, NULL, NULL, &status);
  queue = clCreateCommandQueue(context, *device_list, 0, &status);
  // Create memory buffers on the device for each vector
  clm_pic = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, cls*rws*sizeof(cl_uchar), NULL, &status);
  clm_sh = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, 2*sizeof(cl_float), NULL, &status);
  status = clFinish(queue);
  // Create programs from the kernel source
  prog = clCreateProgramWithSource(context, 1, (const char**)&s_pic2bs, NULL, &status);
  status = clBuildProgram(prog, 1, device_list, NULL, NULL, NULL);
  // Create the OpenCL kernels
  kern = clCreateKernel(prog, "pic2hprof", &status);
  //status = clSetKernelArg(kern, 1, sizeof(cl_mem), (void*)&clm_sh);
  status = clSetKernelArg(kern, 2, sizeof(cl_int), (void*)&cols);
  status = clSetKernelArg(kern, 3, sizeof(cl_int), (void*)&rows);
}

Optzd::~Optzd(){
status = clReleaseKernel(kern);
status = clReleaseProgram(prog);
status = clReleaseMemObject(clm_pic); 
status = clReleaseMemObject(clm_sh); 
status = clReleaseCommandQueue(queue);
status = clReleaseContext(context);
free(platform);
free(device_list);
}

void*
Optzd::eval(uint8_t* t){
  cl_float* sh = (cl_float*)clEnqueueMapBuffer(queue, clm_sh, CL_TRUE, CL_MAP_WRITE, 0, 2*sizeof(cl_float), 0, NULL, NULL, &status);
  //for (int i = 0; i < global_threads; i++) {
  sh[0] = 0;
  sh[1] = 0;
  //}
  // Copy frame to the device
  status = clEnqueueWriteBuffer(queue, clm_pic, CL_FALSE, 0, cols * rows * sizeof(cl_uchar), t, 0, NULL, NULL);
  //status = clEnqueueWriteBuffer(queue, clm_sh, CL_TRUE, 0, global_threads * sizeof(cl_float), sh, 0, NULL, NULL);
  //setting kernels args
  status = clSetKernelArg(kern, 0, sizeof(cl_mem), (void*)&clm_pic);
  status = clSetKernelArg(kern, 1, sizeof(cl_mem), (void*)&clm_sh);
  // Execute the OpenCL kernel on the list
  status = clEnqueueNDRangeKernel(queue, kern, 1, NULL, &global_threads, NULL, 0, NULL, NULL);
  //status = clEnqueueReadBuffer(queue, clm_sh, CL_TRUE, 0, global_threads*sizeof(float), sh, 0, NULL, NULL);
  // Clean up and wait for all the comands to complete.
  status = clFlush(queue);
  status = clFinish(queue);
  //eval-ing BS
  float shb = sh[1], shnb = sh[0];
  // for (int i = 0; i < (global_threads / 2); i++) {
  //   shb += sh[i*2];
  //   shnb += sh[(i*2)+1];
  // }
  clEnqueueUnmapMemObject(queue, clm_sh, sh, 0, NULL, NULL);
  if (shnb == 0) shnb = 4;
  *BS = shb/(shnb);
  return BS;
}

void*
Optzd::eval(std::vector<uint8_t>* t){
  cl_float* sh = (cl_float*)clEnqueueMapBuffer(queue, clm_sh, CL_FALSE, CL_MAP_WRITE, 0, 2*sizeof(cl_float), 0, NULL, NULL, &status);
  uint8_t* pic = reinterpret_cast<uint8_t*>(t->data());
  sh[0] = 0;
  sh[1] = 0;
  // Copy frame to the device
  status = clEnqueueWriteBuffer(queue, clm_pic, CL_TRUE, 0, cols * rows * sizeof(cl_uchar), pic, 0, NULL, NULL);
  //status = clEnqueueWriteBuffer(queue, clm_sh, CL_TRUE, 0, global_threads * sizeof(cl_float), sh, 0, NULL, NULL);
  // setting kernels args
  status = clSetKernelArg(kern, 0, sizeof(cl_mem), (void*)&clm_pic);
  status = clSetKernelArg(kern, 1, sizeof(cl_mem), (void*)&clm_sh);
  // Execute the OpenCL kernel on the list
  status = clEnqueueNDRangeKernel(queue, kern, 1, NULL, &global_threads, NULL, 0, NULL, NULL);
  //status = clEnqueueReadBuffer(queue, clm_sh, CL_TRUE, 0, 2*sizeof(float), sh, 0, NULL, NULL);
  // Clean up and wait for all the comands to complete.
  status = clFlush(queue);
  status = clFinish(queue);
  //eval-ing BS
  float shb = sh[1], shnb = sh[0];
  // for (int i = 0; i < (global_threads / 2); i++) {
  //   shb += sh[i*2];
  //   shnb += sh[(i*2)+1];
  // }
  
  clEnqueueUnmapMemObject(queue, clm_sh, sh, 0, NULL, NULL);
  if (shnb == 0) shnb = 4;
  *BS = shb/(shnb);
  return BS;
}


char*
Optzd::compilerlog(void){
  cl_int logStatus;							
  char * buildLog = NULL;						
  size_t buildLogSize = 0;
  logStatus = clGetProgramBuildInfo(prog, *device_list, CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, &buildLogSize);
  buildLog = new char [buildLogSize];
  logStatus = clGetProgramBuildInfo(prog, *device_list, CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL);
  return buildLog;
}
