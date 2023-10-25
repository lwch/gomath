package tensor

/*
#cgo CXXFLAGS: -mavx -mavx2 -mavx512f -mavx512dq -march=native -O3 -std=c++11
#include "fp16.h"
#include "fp32.h"
*/
import "C"

var debug bool
var fp16ComputeInC bool
var fp32ComputeInC bool

func init() {
	if C.fp16_init() {
		fp16ComputeInC = true
	}
	if C.fp32_init() {
		fp32ComputeInC = true
	}
}
