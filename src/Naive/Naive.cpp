#include "Naive.hpp"

template<typename T>
Naive<T>::Naive(int cls, int rws){
  cols = cls;
  rws = rows;
  BS = new double;
  hDifference = new std::vector<long> (cols);
  hProfile = new std::vector<long> (cols);
}

template<typename T>
Naive<T>::~Naive(){
  delete hDifference;
  delete hProfile;
}

template<typename T>
void*
Naive<T>::eval(std::vector<std::vector<T>> t){
  double Shblock = 0;
  double Shnonblock = 0;
  long block_cnt = 0;
  long nonblock_cnt = 0;
  double* BS = new double;
  double denom;
  long sum;
  for (int i = 0; i < rows; i++) {
    hDifference[0] = std::abs(t[i][0] - t[i][1]);
    for (int j = 0; j < cols; j++) {
      hDifference[j+1] = std::abs(t[i][j] - t[i][j+1]);
      sum = hDifference[j-1] + hDifference[j+1];
      if(!sum) denom = 1;
      else  denom = 0.5 * sum;
      hProfile[j] += (double) hDifference[j] /denom;
    }
  }
  for (int i = 0; i < cols - 1; i++) {
    if (!(i%8)) Shblock += hProfile[i];
    else Shnonblock += hProfile[i];
  }
  block_cnt = cols/8;
  nonblock_cnt = cols - block_cnt;
  if (!Shnonblock) Shnonblock = 4;
  *BS = (Shblock/block_cnt)/(Shnonblock/nonblock_cnt);
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
template class Naive<int>;
