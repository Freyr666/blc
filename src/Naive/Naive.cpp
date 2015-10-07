#include "Naive.hpp"

#include <iostream>

template<typename T>
Naive<T>::Naive(int cls, int rws){
  cols = cls;
  rows = rws;
  BS = new double;
  hProfile = new std::vector<float> (cls/4);
}

template<typename T>
Naive<T>::Naive(Naive &n){
  cols = n.cols;
  rows = n.rows;
  BS = n.BS;
  hProfile = n.hProfile;
}

template<typename T>
Naive<T>::~Naive(){
  delete hProfile;
  delete BS;
}

template<typename T>
void*
Naive<T>::eval(std::vector<T>* t){
  double Shblock = 0;
  double Shnonblock = 0;
  for (int j = 0; j < cols/4; j++) {
    float sum, sub, subNext, subPrev, hacc = 0;
    for (int i = 0; i < rows; i++) {
      sub = (float)abs((*t)[j*4 + i*cols - 1] - (*t)[j*4 + i*cols]);
      subNext = (float)abs((*t)[j*4 + i*cols] - (*t)[j*4 + i*cols + 1]); 
      subPrev = (float)abs((*t)[j*4 + i*cols - 2] - (*t)[j*4 + i*cols - 1]);
      sum = (subNext + subPrev);
      if (sum == 0) sum = 1; 
      else sum = sum / 2; 
      hacc += sub / sum;}
    (*hProfile)[j] = hacc;}
  for (int i = 0; i < cols/4; i++) {
    if (i%2) Shnonblock += (*hProfile)[i];
    else Shblock += (*hProfile)[i];
  }
  if (!Shnonblock) Shnonblock = 4;
  *BS = (Shblock)/(Shnonblock);
  return (void*)BS;
}


template class Naive<uint8_t>;
template class Naive<uint16_t>;
template class Naive<uint32_t>;
template class Naive<uint64_t>;
template class Naive<int8_t>;
template class Naive<int16_t>;
template class Naive<int32_t>;
template class Naive<int64_t>;

