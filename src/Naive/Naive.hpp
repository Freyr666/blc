#ifndef NAIVE_H
#define NAIVE_H

#include <vector>
#include <cmath>

template<typename T>
class Naive{
private:
  int cols;
  int rows;
  double* BS;
  std::vector<float>* hProfile;  
public:
  Naive(int cls, int rws);
  Naive(Naive &n);
  virtual ~Naive();

  void* eval(std::vector<T>* t);

  void* operator()(std::vector<T>* t) { return eval(t);}
};
#endif /* NAIVE_H */
