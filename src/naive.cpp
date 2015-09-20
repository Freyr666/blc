#include "naive.hpp"
#include <iostream>

template<typename T>
fun<T>
get_naive_alg(int cols, int rows){
  fun<T> alg = [cols, rows](std::vector<row<T>> r) -> void* {
    double Shblock = 0;
    double Shnonblock = 0;
    long block_cnt = 0;
    long nonblock_cnt = 0;
    double* BS = new double;
    double denom;
    long sum;
    std::vector<long> hDifference  (cols);
    std::vector<long> hProfile (cols);
    std::cout << "We are in returned f\n";
    for (int i = 0; i < rows; i++) {
      hDifference[0] = std::abs(r[i][0] - r[i][1]);
      for(int j = 0; j < cols - 1 ; j++){
	hDifference[j+1] = std::abs(r[i][j] - r[i][j+1]);
	sum = hDifference[j-1] + hDifference[j+1];
	if (!sum) denom = 1;
	else denom = 0.5*sum;
	hProfile[j] += (double)hDifference[j] / denom;
      }
    }
    for(int i = 0; i < cols - 1 ; i++){
      if(!(i%8)){
	Shblock += hProfile[i];
	block_cnt++;
      }
      else{
	Shnonblock += hProfile[i];
	nonblock_cnt++;
      }
    }
    if(!Shnonblock) Shnonblock = 4;
    std::cout << "And all seems ok\n";
    *BS = (Shblock/block_cnt)/(Shnonblock/nonblock_cnt);
    std::cout << "BS is " << *BS << "\n";
    std::cout << "sh is " << Shblock << " shnb is " << Shnonblock << "\n";
    std::cout << "cnt is " << block_cnt << " cntnb is " << nonblock_cnt << "\n";
    return BS;
  };
  return alg;
}

template fun<uint8_t> get_naive_alg(int cols, int rows);
template fun<uint16_t> get_naive_alg(int cols, int rows);
template fun<uint32_t> get_naive_alg(int cols, int rows);
template fun<uint64_t> get_naive_alg(int cols, int rows);
template fun<int8_t> get_naive_alg(int cols, int rows);
template fun<int16_t> get_naive_alg(int cols, int rows);
template fun<int32_t> get_naive_alg(int cols, int rows);
template fun<int64_t> get_naive_alg(int cols, int rows);
