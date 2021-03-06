#include <vector>
#include <iostream>
#include <algorithm>
#include "Matrix.hpp"
#include "Pic2Mat.hpp"
//#include "./Naive/Naive.hpp"
//#include "./opencl++/Optzd.hpp"
#include "./Alg/Alg.hpp"
//timestamps for testing
#include <sys/time.h>
//multi threading test
#include <thread>

// void
// in_thread_opt(std::vector<uint8_t>* pic, int n, Optzd* fun){
//   double rv = *(double*)(*fun)(pic);
//   std::cout << "Thread " << n << " " << rv << "\n";
// }

// void
// in_thread(std::vector<uint8_t>* pic, int n, Naive<uint8_t>* fun){
//   double rv = *(double*)(*fun)(pic);
//   // std::cout << "Thread " << n << " " << rv << "\n";
// }

int
main(int argc, char **argv){
  if (argc < 2){
    std::cout << "Usage: prog ./image\n";
    return 0;
  }
  
  struct timeval tvb, tva;
  Matrix<uint8_t>* blc[argc-1];
  //  Naive<uint8_t>* f[argc-1];
  float* perc;
  // Optzd* fopt[argc-1];
  // std::thread thread_list[argc-1];
  for (int i = 0; i < argc - 1; i++) {
    blc[i] = Pic2Mat<uint8_t>(argv[i+1]);
    // f[i] = new Naive<uint8_t> (blc[i]->get_cols_num(), blc[i]->get_rows_num());
    // fopt[i] = new Optzd  (blc[i]->get_cols_num(), blc[i]->get_rows_num(), 16, 1);
  }
  Alg alg (blc[0]->get_cols_num(), blc[0]->get_rows_num(), 16, 1, 16, 16);
  gettimeofday(&tvb,NULL);
  // for (int i = 0; i < argc-1; i++) {
  //   thread_list[i] = std::thread(in_thread_opt, (blc[i]->val()), i, fopt[i]);
  // }
  // for (int i = 0; i < argc-1; i++) {
  //   thread_list[i].join();
  // }
  perc = (float*)alg(blc[0]->val(), blc[1]->val());
  gettimeofday(&tva,NULL);
  
  std::cout << "black: " << perc[0] << "\tSame: " << perc[1] << "\tAvg diff: " << perc[2] << "\tAvg int: " << perc[3] << "\tBlcns: " << perc[4] << " Opt in " << " " << tva.tv_usec - tvb.tv_usec << " us\n" << alg.compilerlog();

  // gettimeofday(&tvb,NULL);
  // for (int i = 0; i < argc-1; i++) {
  //   thread_list[i] = std::thread(in_thread, (blc[i]->val()), i, f[i]);
  // }
  // for (int i = 0; i < argc-1; i++) {
  //   thread_list[i].join();
  // }
  // gettimeofday(&tva,NULL);
  
  // std::cout << "Not opt " << tva.tv_usec - tvb.tv_usec << " us\n";

  return 0;
}
