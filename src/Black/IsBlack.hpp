#ifndef ISBLACK_H
#define ISBLACK_H

#include <CL/cl.h>
#include <vector>
#include <cmath>

class IsBlack{
private:
  int cols, rows, platform_n;
  uint8_t val;
  float* percentage;
  int* result;
  
  cl_mem clm_pic, clm_res;
  cl_platform_id * platform = NULL;
  cl_device_id * device_list = NULL;
  cl_context context;
  cl_command_queue queue;
  cl_uint n_platforms, n_devices;
  size_t global_threads, threads;
  cl_program prog;
  cl_kernel kern;
  cl_int status;
public:
  IsBlack(int cls, int rws, int thrds, int pltfrm, uint8_t v);
  virtual ~IsBlack();

  void* eval(uint8_t* t);
  void* eval(std::vector<uint8_t>* t);
  void* operator()(uint8_t* t) {return eval(t);}
  void* operator()(std::vector<uint8_t>* t) {return eval(t);}

  char* compilerlog(void);
};

#endif /* ISBLACK_H */
