#ifndef OPTZD_H
#define OPTZD_H

#include <CL/cl.h>
#include <vector>
#include <cmath>

#include "kernels.h"

class Optzd{
private:
  int cols, rows, threads;
  float* sh;
  double* BS;

  cl_mem clm_pic, clm_res;
  cl_platform_id * platform = NULL;
  cl_device_id * device_list = NULL;
  cl_context context;
  cl_command_queue queue;
  cl_uint n_platforms, n_devices;
  size_t global_threads;
  cl_program prog;
  cl_kernel kern;
  cl_int status;
public:
  Optzd(int cls, int rws, int thrds, int pltfrm);
  virtual ~Optzd();

  void* eval(uint8_t* t);

  void* eval(std::vector<uint8_t>* t);

  void* operator()(std::vector<uint8_t>* t) { return eval(t);}

  void* operator()(uint8_t* t) { return eval(t);}

  char* compiler_log(void);
};
#endif /* OPTZD_H */
