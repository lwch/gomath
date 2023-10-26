package ctensor

// #include "fp16/fp16.h"
import "C"

func (*CTensor) FP16DotVector(x, w []uint16) uint16 {
	return uint16(C.fp16_dot_vector(
		(*C.uint16_t)(&x[0]),
		(*C.uint16_t)(&w[0]),
		C.int64_t(len(x)),
	))
}

func (*CTensor) FP16Mul(x, w, y []uint16) {
	C.fp16_mul_vector(
		(*C.uint16_t)(&x[0]),
		(*C.uint16_t)(&w[0]),
		(*C.uint16_t)(&y[0]),
		C.int64_t(len(x)),
	)
}

func (*CTensor) FP16MulScalar(x []uint16, w uint16, y []uint16) {
	C.fp16_mul_scalar(
		(*C.uint16_t)(&x[0]),
		C.uint16_t(w),
		(*C.uint16_t)(&y[0]),
		C.int64_t(len(x)),
	)
}
