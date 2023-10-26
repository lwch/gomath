package tensor

/*
#cgo CXXFLAGS: -mavx -mavx2 -mavx512f -mavx512dq -march=native -O3 -std=c++11 -Wno-ignored-attributes
#include "tensor.h"
*/
import "C"

var debug bool
var cCompute bool

func init() {
	if C.tensor_init() {
		cCompute = true
	}
}
