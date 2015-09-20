project(blk CXX)

add_compile_options("`Magick++-config --cppflags --cxxflags --ldflags --libs`")

add_executable(blc_test main.cpp)


