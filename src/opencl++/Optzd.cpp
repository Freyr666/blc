#include "Optzd.hpp"

#include <iostream>

template<typename T>
Optzd<T>::Optzd(int cls, int rws, int l_mem){
  //vals init
  cols = cls;
  rows = rws;
  lmem = l_mem;
  fsize = cls*rws;
  cnt_blc = cls/8;
  cnt_nblc = cls - cls/8;
  BS = new double;
  //debug
  ddiff = new int [fsize];
  //END_DEBUG
  //opencl init
  //Set up the Platform
  clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
  platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id)*num_platforms);
  clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);
  //Get the devices list and choose the device you want to run on
  clStatus = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
  device_list = (cl_device_id *) malloc(sizeof(cl_device_id)*num_devices);
  clStatus = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, num_devices, device_list, NULL);
  // Create one OpenCL context for each device in the platform
  context = clCreateContext( NULL, num_devices, device_list, NULL, NULL, &clStatus);
  // Create a command queue
  command_queue = clCreateCommandQueue(context, *device_list, 0, &clStatus);
  // Create memory buffers on the device for each vector
  cl_frame = clCreateBuffer(context, CL_MEM_READ_ONLY, fsize * sizeof(T), NULL, &clStatus); //frame buf
  cl_fs = clCreateBuffer(context, CL_MEM_READ_ONLY, fsize * sizeof(T), NULL, &clStatus);
  cl_hdif = clCreateBuffer(context, CL_MEM_READ_WRITE, fsize * sizeof(int), NULL, &clStatus); //difference buf
  cl_prof = clCreateBuffer(context, CL_MEM_READ_WRITE, cols * sizeof(float), NULL, &clStatus); //
  cl_shblc = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, &clStatus); //Sh on edges
  cl_shnblc = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, &clStatus); //Sh
  // Create programs from the kernel source
  p_frame2dif = clCreateProgramWithSource(context, 1, (const char **)&frame2dif, NULL, &clStatus);
  // p_dif2prof = clCreateProgramWithSource(context, 1, (const char **)&dif2prof, NULL, &clStatus);
  //p_sum2sh = clCreateProgramWithSource(context, 1, (const char **)&sum2sh, NULL, &clStatus);
  // Build the program
  clStatus = clBuildProgram(p_dif2prof, 1, device_list, NULL, NULL, NULL);
  //clStatus = clBuildProgram(p_frame2dif, 1, device_list, NULL, NULL, NULL);
  //clStatus = clBuildProgram(p_sum2sh, 1, device_list, NULL, NULL, NULL);
  // Create the OpenCL kernels
  k_dif2prof = clCreateKernel(p_dif2prof, "dif2prof_kernel", &clStatus);
  //k_frame2dif = clCreateKernel(p_frame2dif, "frame2dif_kernel", &clStatus);
  //k_sum2sh = clCreateKernel(p_sum2sh, "sum2sh_kernel", &clStatus);
}

template<typename T>
Optzd<T>::~Optzd(){
  /*
  clStatus = clReleaseKernel(kernel);
clStatus = clReleaseProgram(program);
clStatus = clReleaseMemObject(A_clmem);
clStatus = clReleaseMemObject(B_clmem);
clStatus = clReleaseMemObject(C_clmem);
clStatus = clReleaseCommandQueue(command_queue);
clStatus = clReleaseContext(context);
free(A);
free(B);
free(C);
free(platforms);
free(device_list);*/
}

template<typename T>
void*
Optzd<T>::eval(std::vector<T>* t){
  fs = (int*)t + 1;
  // Copy frame to the device
  clStatus = clEnqueueWriteBuffer(command_queue, cl_frame, CL_TRUE, 0, fsize * sizeof(int), (T*)t, 0, NULL, NULL);
  clStatus = clEnqueueWriteBuffer(command_queue, cl_fs, CL_TRUE, 0, fsize * sizeof(int), fs, 0, NULL, NULL);
  //setting kernels args
  clStatus = clSetKernelArg(k_frame2dif, 0, sizeof(cl_mem), (void *)&cl_frame);
    clStatus = clSetKernelArg(k_frame2dif, 1, sizeof(cl_mem), (void *)&cl_fs);
  clStatus = clSetKernelArg(k_frame2dif, 2, sizeof(cl_mem),  (void *)&cl_hdif);
  clStatus = clSetKernelArg(k_frame2dif, 2, sizeof(cl_int),  (void *)&fsize);
//clStatus = clSetKernelArg(k_frame2dif, 2, sizeof(int),   (void *)&fsize);
  //dif2prof
  /*clStatus = clSetKernelArg(k_dif2prof, 0, sizeof(cl_mem),   (void *)&cl_hdif);
  clStatus = clSetKernelArg(k_dif2prof, 1, sizeof(cl_mem), (void *)&cl_prof);
  clStatus = clSetKernelArg(k_dif2prof, 2, sizeof(int),  (void *)&cols);  
  clStatus = clSetKernelArg(k_dif2prof, 3, sizeof(int),  (void *)&fsize);*/
  //sum2sh
  /* clStatus = clSetKernelArg(k_sum2sh, 0, sizeof(cl_mem),
			    (void *)&cl_prof);
  clStatus = clSetKernelArg(k_sum2sh, 1, sizeof(cl_mem),
			    (void *)&cl_shnblc);
  clStatus = clSetKernelArg(k_sum2sh, 2, sizeof(cl_mem),
			    (void *)&cl_shblc);  
  clStatus = clSetKernelArg(k_sum2sh, 3, sizeof(int),
  (void *)&fsize);*/
  // Execute the OpenCL kernel on the list
  size_t global_size = fsize; // Process the entire lists
  size_t local_size = lmem;
  // Process one item at a time
  clStatus = clEnqueueNDRangeKernel(command_queue, k_dif2prof, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
  //clStatus = clEnqueueNDRangeKernel(command_queue, k_frame2dif, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
  //clStatus = clEnqueueNDRangeKernel(command_queue, k_sum2sh, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
// Read the cl memory C_clmem on device to the host variable C
  //clStatus = clEnqueueReadBuffer(command_queue, cl_shnblc, CL_TRUE, 0, sizeof(float), &Shnblc, 0, NULL, NULL);
  //clStatus = clEnqueueReadBuffer(command_queue, cl_shblc, CL_TRUE, 0, sizeof(float), &Shblc, 0, NULL, NULL);
  clStatus = clEnqueueReadBuffer(command_queue, cl_shblc, CL_TRUE, 0, fsize*sizeof(int), &ddiff, 0, NULL, NULL);
  // Clean up and wait for all the comands to complete.
  clStatus = clFlush(command_queue);
  clStatus = clFinish(command_queue);
  //eval-ing BS
  for (int i = 0; i < fsize; i++) {
    std::cout << ddiff[i] << " ";
  }

  std::cout << Shblc << " " << Shnblc << "\n";
  *BS = (Shblc/cnt_blc)/(Shnblc/cnt_nblc);
  return BS;
}

/*
template class Naive<uint8_t>;
template class Naive<uint16_t>;
template class Naive<uint32_t>;
template class Naive<uint64_t>;
template class Naive<int8_t>;
template class Naive<int16_t>;
template class Naive<int32_t>;
template class Naive<int64_t>;
*/
template class Optzd<int>;
