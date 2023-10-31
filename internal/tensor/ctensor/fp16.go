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

func (*CTensor) FP16MulScalar(x uint16, w, y []uint16) {
	C.fp16_mul_scalar(
		C.uint16_t(x),
		(*C.uint16_t)(&w[0]),
		(*C.uint16_t)(&y[0]),
		C.int64_t(len(w)),
	)
}

func (*CTensor) FP16MulVector(x, w, y []uint16) {
	C.fp16_mul_vector(
		(*C.uint16_t)(&x[0]),
		(*C.uint16_t)(&w[0]),
		(*C.uint16_t)(&y[0]),
		C.int64_t(len(x)),
	)
}

func (*CTensor) FP16DivScalar(x uint16, w, y []uint16) {
	C.fp16_div_scalar(
		C.uint16_t(x),
		(*C.uint16_t)(&w[0]),
		(*C.uint16_t)(&y[0]),
		C.int64_t(len(w)),
	)
}

func (*CTensor) FP16DivVector(x, w, y []uint16) {
	C.fp16_div_vector(
		(*C.uint16_t)(&x[0]),
		(*C.uint16_t)(&w[0]),
		(*C.uint16_t)(&y[0]),
		C.int64_t(len(x)),
	)
}

func (*CTensor) FP16AddScalar(x uint16, w, y []uint16) {
	C.fp16_add_scalar(
		C.uint16_t(x),
		(*C.uint16_t)(&w[0]),
		(*C.uint16_t)(&y[0]),
		C.int64_t(len(w)),
	)
}

func (*CTensor) FP16AddVector(x, w, y []uint16) {
	C.fp16_add_vector(
		(*C.uint16_t)(&x[0]),
		(*C.uint16_t)(&w[0]),
		(*C.uint16_t)(&y[0]),
		C.int64_t(len(x)),
	)
}
