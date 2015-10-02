#include "Optzd.hpp"

const char * s_pic2bs =
  //"#pragma OPENCL EXTENSION cl_khr_fp64 : enable  \n"
  "__kernel void pic2hprof(\n"  //calcs hprof for each row
  "         __global const uchar* pic,\n"
  "         __global float* result,\n"
  "         int hsize, int vsize)\n"
  "{    \n"
  "  size_t globalId = get_global_id(0); \n"
  "  size_t groupSize = get_num_groups(0); \n"
  "  size_t localSize = get_local_size(0); \n"
  "  size_t groupId = get_group_id(0); \n"
  "  size_t localId = get_local_id(0); \n"
  "  size_t globalSize = get_global_size(0); \n"
  "  size_t binSize = globalSize/localSize;\n"
  "  float sum, sub, subNext, subPrev; \n"
  "  float hacc = 0;\n"
  // "  if ((globalId < 2) || (globalId > hsize - 1) || (globalId > globalSize - 1)) return; \n"
  "  for (int i = 0; i < vsize; ++i){ \n"
  "         sub = (float)abs(pic[globalId*4 + i*hsize - 1] - pic[globalId*4 + i*hsize]);\n"
  "         subNext = (float)abs(pic[globalId*4 + i*hsize] - pic[globalId*4 + i*hsize + 1]);\n" 
  "         subPrev = (float)abs(pic[globalId*4 + i*hsize - 2] - pic[globalId*4 + i*hsize - 1]);\n"
  "         sum = (float)(subNext + subPrev);\n"
  "         if (sum == 0) sum = 1; \n"
  "         else sum = 0.5*sum; \n"
  "         hacc += sub / sum;} \n"
  "  if (globalId % 2) result[0] += hacc;\n"
  "  else result[1] += hacc;\n"
  "}\n";

#include <iostream>

Optzd::Optzd(int cls, int rws, int thrds, int pltfrm){
  //vals init
  cols = cls;
  rows = rws;
  threads = thrds;
  global_threads = cls/4;
  BS = new double;
  sh = new float[2];
  //opencl init
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
  clm_pic = clCreateBuffer(context, CL_MEM_READ_ONLY, cls*rws*sizeof(cl_uchar), NULL, &status);
  clm_res = clCreateBuffer(context, CL_MEM_READ_WRITE, 2*sizeof(cl_float), NULL, &status);
  status = clFinish(queue);
  // Create programs from the kernel source
  prog = clCreateProgramWithSource(context, 1, (const char**)&s_pic2bs, NULL, &status);
  status = clBuildProgram(prog, 1, device_list, NULL, NULL, NULL);
  // Create the OpenCL kernels
  kern = clCreateKernel(prog, "pic2hprof", &status);
  status = clSetKernelArg(kern, 1, sizeof(cl_mem), (void*)&clm_res);
  status = clSetKernelArg(kern, 2, sizeof(cl_int), (void*)&cols);
  status = clSetKernelArg(kern, 3, sizeof(cl_int), (void*)&rows);
}

Optzd::~Optzd(){
status = clReleaseKernel(kern);
status = clReleaseProgram(prog);
status = clReleaseMemObject(clm_pic);
status = clReleaseMemObject(clm_res);
status = clReleaseCommandQueue(queue);
status = clReleaseContext(context);
free(platform);
free(device_list);
}

void*
Optzd::eval(uint8_t* t){
  // Copy frame to the device
  status = clEnqueueWriteBuffer(queue, clm_pic, CL_TRUE, 0, cols * rows * sizeof(cl_uchar), t, 0, NULL, NULL);
  //setting kernels args
  status = clSetKernelArg(kern, 0, sizeof(cl_mem), (void*)&clm_pic);
  // Execute the OpenCL kernel on the list
  status = clEnqueueNDRangeKernel(queue, kern, 1, NULL, &global_threads, NULL, 0, NULL, NULL);
  status = clEnqueueReadBuffer(queue, clm_res, CL_TRUE, 0, 2*sizeof(float), sh, 0, NULL, NULL);
  // Clean up and wait for all the comands to complete.
  status = clFlush(queue);
  status = clFinish(queue);
  //eval-ing BS
  *BS = (sh[1])/(sh[0]);
  return BS;
}

void*
Optzd::eval(std::vector<uint8_t>* t){
  uint8_t* pic = reinterpret_cast<uint8_t*>(t->data());
  // Copy frame to the device
  status = clEnqueueWriteBuffer(queue, clm_pic, CL_TRUE, 0, cols * rows * sizeof(cl_uchar), pic, 0, NULL, NULL);
  // setting kernels args
  status = clSetKernelArg(kern, 0, sizeof(cl_mem), (void*)&clm_pic);
  // Execute the OpenCL kernel on the list
  status = clEnqueueNDRangeKernel(queue, kern, 1, NULL, &global_threads, NULL, 0, NULL, NULL);
  status = clEnqueueReadBuffer(queue, clm_res, CL_TRUE, 0, 2*sizeof(float), sh, 0, NULL, NULL);
  // Clean up and wait for all the comands to complete.
  status = clFlush(queue);
  status = clFinish(queue);
  //eval-ing BS
  *BS = (sh[1])/(sh[0]);
  return BS;
}
