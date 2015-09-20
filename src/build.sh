#!/bin/bash

g++  main.cpp Matrix.cpp Pic2Mat.cpp -o y --std=c++11 -g -fopenmp -DMAGICKCORE_HDRI_ENABLE=0 -DMAGICKCORE_QUANTUM_DEPTH=16 -fopenmp -DMAGICKCORE_HDRI_ENABLE=0 -DMAGICKCORE_QUANTUM_DEPTH=16 -fopenmp -DMAGICKCORE_HDRI_ENABLE=0 -DMAGICKCORE_QUANTUM_DEPTH=16 -I/usr/include/ImageMagick-6 
-fopenmp -DMAGICKCORE_HDRI_ENABLE=0 -DMAGICKCORE_QUANTUM_DEPTH=16 -fopenmp -DMAGICKCORE_HDRI_ENABLE=0 -DMAGICKCORE_QUANTUM_DEPTH=16 -fopenmp -DMAGICKCORE_HDRI_ENABLE=0 -DMAGICKCORE_QUANTUM_DEPTH=16 -I/usr/include/ImageMagick-6 
-lMagick++-6.Q16 -lMagickWand-6.Q16 -lMagickCore-6.Q16 
-lMagick++-6.Q16 -lMagickWand-6.Q16 -lMagickCore-6.Q16 
