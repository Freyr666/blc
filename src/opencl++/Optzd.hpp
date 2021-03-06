#ifndef OPTZD_H
#define OPTZD_H

#include <CL/cl.h>
#include <vector>
#include <cmath>

class Optzd{
private:
  int cols, rows;
  float* sh;
  double* BS;

  cl_mem clm_pic, clm_sh, clm_shb;
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
  Optzd(int cls, int rws, int thrds, int pltfrm);
  virtual ~Optzd();

  void* eval(uint8_t* t);

  void* eval(std::vector<uint8_t>* t);

  void* operator()(std::vector<uint8_t>* t) { return eval(t);}

  void* operator()(uint8_t* t) { return eval(t);}

  char* compilerlog(void);

  int getstatus(void) const {return status;}
};
#endif /* OPTZD_H */
