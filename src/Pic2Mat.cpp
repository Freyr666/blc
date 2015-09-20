#include "Pic2Mat.hpp"

template<typename T>
Matrix<T>* Pic2Mat(const char* src){
  Magick::InitializeMagick(src);
  Magick::Image img;
  //  try {
    img.read(src);
    img.colorspaceType(Magick::YCbCrColorspace);
    //    Matrix<T>* mat = Magick2Mat<T>(&img);
    //  } catch ( Exception &error_ ) {
    //    throw &error_;
    //  }
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
      T c = color.redQuantum() / range;
      mat->set_el_m(j, i, c);      
    }
  }
  return mat;
}

template Matrix<int8_t>* Pic2Mat(const char* src);
template Matrix<int8_t>* Magick2Mat(Magick::Image * img);
template Matrix<int16_t>* Pic2Mat(const char* src);
template Matrix<int16_t>* Magick2Mat(Magick::Image * img);
template Matrix<int32_t>* Pic2Mat(const char* src);
template Matrix<int32_t>* Magick2Mat(Magick::Image * img);
template Matrix<int64_t>* Pic2Mat(const char* src);
template Matrix<int64_t>* Magick2Mat(Magick::Image * img);
