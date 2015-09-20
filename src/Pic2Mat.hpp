#ifndef PIC2MAT_H
#define PIC2MAT_H

#include <Magick++.h>
#include <vector>

#include "Matrix.hpp"

template<typename T>
Matrix<T>* Pic2Mat(const char* src);

template<typename T>
Matrix<T>* Magick2Mat(Magick::Image * img);

#endif /* PIC2MAT_H */
