#include "Pic2Mat.hpp"
#include <iostream>

template<typename T>
Matrix<T>* Pic2Mat(const char* src){
  Magick::InitializeMagick(src);
  Magick::Image img;
  img.read(src);
  Matrix<T>* mat = Magick2Mat<T>(&img);
  return mat;
}

template<typename T>
Matrix<T>* Magick2Mat(Magick::Image * img){
  Matrix<T>* mat = new Matrix<T>(img->columns(), img->rows());
  int range = pow(2, img->modulusDepth());
  MagickCore::PixelPacket *pixels = img->getPixels((ssize_t)0, (ssize_t)0, img->columns(), img->rows());
  for (int i = 0; i < img->rows() - 1; i++) {
    for (int j = 0; j < img->columns() - 1; j++) {
      Magick::Color color = pixels[img->columns() * i + j];
      uint8_t rc = color.redQuantum() /range;
      uint8_t bc = color.blueQuantum() /range;
      uint8_t gc = color.greenQuantum() /range;
      T c = 16 + (65.738*rc/256) + (25.064*bc/256) + (129.057*gc/256);
      mat->set_el_m(j, i, c);      
    }
  }
  return mat;
}

template Matrix<uint8_t>* Pic2Mat(const char* src);
template Matrix<uint8_t>* Magick2Mat(Magick::Image * img);
template Matrix<uint16_t>* Pic2Mat(const char* src);
template Matrix<uint16_t>* Magick2Mat(Magick::Image * img);
template Matrix<uint32_t>* Pic2Mat(const char* src);
template Matrix<uint32_t>* Magick2Mat(Magick::Image * img);
template Matrix<uint64_t>* Pic2Mat(const char* src);
template Matrix<uint64_t>* Magick2Mat(Magick::Image * img);
template Matrix<int8_t>* Pic2Mat(const char* src);
template Matrix<int8_t>* Magick2Mat(Magick::Image * img);
template Matrix<int16_t>* Pic2Mat(const char* src);
template Matrix<int16_t>* Magick2Mat(Magick::Image * img);
template Matrix<int32_t>* Pic2Mat(const char* src);
template Matrix<int32_t>* Magick2Mat(Magick::Image * img);
template Matrix<int64_t>* Pic2Mat(const char* src);
template Matrix<int64_t>* Magick2Mat(Magick::Image * img);
