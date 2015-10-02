#include <iostream>
#include <vector>
#include <CL/cl.h>

#include "../Matrix.hpp"
#include "../Pic2Mat.hpp"

//timestamps for testing
#include <sys/time.h>
//

#define DIV(X,Y)((X%Y)? X/Y+1 : X/Y)

#define THREADS 16

const char * s_pic2bs =
  //"#pragma OPENCL EXTENSION cl_khr_fp64 : enable  \n"
  "__kernel void pic2hprof(\n"  //calcs hprof for each row
  "         __global const int* pic,\n"
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

void*
pic2bs(int h, int v, int* pic, float* res){
  //ocl init
  cl_platform_id * platform = NULL;
  cl_device_id * device_list = NULL;
  cl_context context;
  cl_command_queue queue;
  cl_uint n_platforms;
  cl_uint n_devices;
  cl_program prog;
  cl_kernel kern;
  cl_int status;
  struct timeval tvb, tva;

  //int* buf = new int [h];
  //p i
  status = clGetPlatformIDs(0, NULL, &n_platforms);
  platform = new cl_platform_id[n_platforms];
  status = clGetPlatformIDs(n_platforms, platform, NULL);
  //d i
  status = clGetDeviceIDs(platform[1], CL_DEVICE_TYPE_GPU, 0, NULL, &n_devices);
  device_list = new cl_device_id [n_devices];
  status = clGetDeviceIDs(platform[1], CL_DEVICE_TYPE_GPU, n_devices, device_list, NULL);
  //c i
  context = clCreateContext(NULL, n_devices, device_list, NULL, NULL, &status);
  queue = clCreateCommandQueue(context, *device_list, 0, &status);
  //m i
  cl_mem clm_pic = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, v*h*sizeof(cl_int), pic, &status);
  cl_mem clm_res = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 2*sizeof(cl_float), res, &status);
  //cping
  status = clFinish(queue);
  //  status = clEnqueueWriteBuffer(queue, clm_pic, CL_TRUE, 0, h*v*sizeof(cl_int), pic, 0, NULL, NULL);
  // status = clEnqueueWriteBuffer(queue, clm_href, CL_TRUE, 0, h*sizeof(cl_float), href, 0, NULL, NULL);
  //
  prog = clCreateProgramWithSource(context, 1, (const char**)&s_pic2bs, NULL, &status);
  status |= clBuildProgram(prog, 1, device_list, NULL, NULL, NULL);
  if (status){
    cl_int logStatus;							
    char * buildLog = NULL;						
    size_t buildLogSize = 0;
    logStatus = clGetProgramBuildInfo(prog, *device_list, CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, &buildLogSize);
    buildLog = new char [buildLogSize];
    logStatus = clGetProgramBuildInfo(prog, *device_list, CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL);
    std::cout << "Here comes the compilation log\n";
    std::cout << buildLog;
    exit(0);
  }
  kern = clCreateKernel(prog, "pic2hprof", &status);
  //s a
  gettimeofday(&tvb,NULL);
  status = clSetKernelArg(kern, 0, sizeof(cl_mem), (void*)&clm_pic);
  status = clSetKernelArg(kern, 1, sizeof(cl_mem), (void*)&clm_res);
  status = clSetKernelArg(kern, 2, sizeof(cl_int), (void*)&h);
  status = clSetKernelArg(kern, 3, sizeof(cl_int), (void*)&v);
  //
  size_t globalThreads = h/4;
  size_t localThreads = THREADS;
  //
  status = clEnqueueNDRangeKernel(queue, kern, 1, NULL, &globalThreads, NULL, 0, NULL, NULL);
  // result
  status = clEnqueueReadBuffer(queue, clm_res, CL_TRUE, 0, 2*sizeof(float), res, 0, NULL, NULL);
  // flush
  status = clFlush(queue);
  status = clFinish(queue);
  gettimeofday(&tva,NULL);
  std::cout << "Kernel t " << tva.tv_usec - tvb.tv_usec << " us\n";
  //
  status = clReleaseKernel(kern);
  status = clReleaseProgram(prog);
  status = clReleaseMemObject(clm_res);
  status = clReleaseMemObject(clm_pic);
  status = clRetainCommandQueue(queue);
  status = clReleaseContext(context);

  delete[] platform;
  delete[] device_list;
  return 0;
}

int
main(int argc, char *argv[]){
  if (argc < 2) return 0;
  struct timeval tvb, tva;
  Matrix<int>* blc = Pic2Mat<int>(argv[1]);
  float Shblock = 0;
  float Shnonblock = 0;
  long block_cnt = 0;
  long nonblock_cnt = 0;
  float BS;
  int v = blc->get_rows_num();
  int h = blc->get_cols_num();
  float* sh = new float[2];
  sh[0] = 0;
  sh[1] = 0;
  int *pic = reinterpret_cast<int*>(blc->val()->data());
  pic2bs(h, v, pic, sh);
  block_cnt = h/8;
  nonblock_cnt = h - block_cnt;
  if (!sh[0]) sh[0] = 4;
  BS = (sh[1])/(sh[0]);
  std::cout << BS << " -- result\n";
  return 0;
}
