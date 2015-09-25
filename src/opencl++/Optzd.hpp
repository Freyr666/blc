#ifndef OPTZD_H
#define OPTZD_H

#include <CL/cl.h>
#include <vector>
#include <cmath>

#include "kernels.h"

template<typename T>
class Optzd{
private:
  int cols, rows, fsize, lmem;
  int cnt_blc, cnt_nblc;
  float Shblc, Shnblc;
  double* BS;
  int* ddiff, *fs;
  cl_mem cl_frame, cl_hdif, cl_prof, cl_shblc, cl_shnblc, cl_fs;
  //cl_spec
  cl_int clStatus;
  cl_platform_id * platforms;
  cl_uint num_platforms;
  cl_device_id* device_list;
  cl_uint num_devices;
  cl_context context;
  cl_command_queue command_queue;
  cl_program p_frame2dif, p_dif2prof, p_sum2sh;
  cl_kernel k_frame2dif, k_dif2prof, k_sum2sh;
public:
  Optzd(int cls, int rws, int l_mem);
  virtual ~Optzd();

  void* eval(std::vector<T>* t);

  void* operator()(std::vector<T>* t) { return eval(t);}
};
#endif /* OPTZD_H */
