#ifndef ISBLACK_H
#define ISBLACK_H

#include <CL/cl.h>
#include <vector>
#include <cmath>

class IsBlack{
private:
  int cols, rows, platform_n;
  uint8_t valbk, valsm;
  float* percentage;
  
  cl_mem clm_pic, clm_picprev, clm_res;
  cl_platform_id * platform = NULL;
  cl_device_id * device_list = NULL;
  cl_context context;
  cl_command_queue queue;
  cl_uint n_platforms, n_devices;
  size_t global_threads, threads;
  cl_program prog;
  cl_kernel kern;
  cl_int status, exit_status;
public:
  IsBlack(int cls, int rws, int thrds, int pltfrm, uint8_t vbk, uint8_t vsm);
  virtual ~IsBlack();

  void* eval(uint8_t* t, uint8_t* tp);
  void* eval(std::vector<uint8_t>* t, std::vector<uint8_t>* tp);
  void* operator()(uint8_t* t, uint8_t* tp) {return eval(t, tp);}
  void* operator()(std::vector<uint8_t>* t, std::vector<uint8_t>* tp) {return eval(t, tp);}

  char* compilerlog(void);
};

#endif /* ISBLACK_H */
