package tensor

/*
#cgo CXXFLAGS: -mavx -mavx2 -mavx512f -mavx512dq
#include "tensor.h"
*/
import "C"

func init() {
	C.tensor_init()
}
