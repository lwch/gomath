package tensor

// #include "tensor.h"
import "C"

func init() {
	C.tensor_init()
}
