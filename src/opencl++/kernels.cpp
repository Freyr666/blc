#include "kernels.h"
/*
const char* frame2dif = "__kernel void frame2dif \
(__global int* frame,\
 __global int* frames,\
 __global int* dif,\
 int fin)\
{int index = get_global_id(0);\
if (index < fin) {\
dif[index] = abs_diff(frame[index], frame[index-1]);}}";
*/
const char* frame2dif = "__kernel void frame2dif \
(__global int* frame,\
 __global int* frames,\
 __global int* dif,\
 int fin)\
{int index = get_global_id(0);\
if (index < fin) {\
dif[index] = frames[index] - frame[index]}";

const char* dif2prof = "__kernel void dif2prof \
(__global float* dif,\
 __global float* prof,\
 int cols,\
 int fin)\
{int index = get_global_id(0);\
if (index >= fin) return; \
if (index = 0) sum = abs_diff(dif[index], dif[index+1]);\
barrier(CLK_LOCAL_MEM_FENCE);\
else sum = abs_diff(dif[index - 1], dif[index + 1]);\
if (sum == 0) sum = 1; \
else sum = sum * 0.5; \
prof[index % cols] += diff[index]/sum;}";

const char* sum2sh = "__kernel void sum \
(__global float* prof,\
 __global float sh,\
 __global float shb,\
 int fin)\
{int index = get_global_id(0);\
if (index >= fin) return; \
if (index % 8) sh += prof[index];\
else shb += prof[index];}";
