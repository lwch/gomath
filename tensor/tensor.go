package tensor

/*
#cgo CXXFLAGS: -mavx -mavx2 -mavx512f -mavx512dq -march=native -O3 -std=c++11
#include "bf16.h"
#include "f32.h"
*/
import "C"

func init() {
	C.bf16_init()
	C.f32_init()
}
