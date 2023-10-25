package tensor

/*
#cgo CXXFLAGS: -mavx -mavx2 -mavx512f -mavx512dq -march=native -O3 -std=c++11
#include "f16.h"
#include "f32.h"
*/
import "C"

var debug bool
var f16ComputeInC bool
var f32ComputeInC bool

func init() {
	if C.f16_init() {
		f16ComputeInC = true
	}
	if C.f32_init() {
		f32ComputeInC = true
	}
}
