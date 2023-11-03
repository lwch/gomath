package ctensor

// #include "fp16/fp16.h"
import "C"

func (*CTensor) FP16Dot(x, w []uint16) uint16 {
	return uint16(C.fp16_dot(
		(*C.uint16_t)(&x[0]),
		(*C.uint16_t)(&w[0]),
		C.int64_t(len(x)),
	))
}

func (*CTensor) FP16Mul(x, w, y []uint16) {
	C.fp16_mul(
		(*C.uint16_t)(&x[0]),
		(*C.uint16_t)(&w[0]),
		(*C.uint16_t)(&y[0]),
		C.int64_t(len(x)),
	)
}

func (*CTensor) FP16ScalarMul(x uint16, w, y []uint16) {
	C.fp16_scalar_mul(
		C.uint16_t(x),
		(*C.uint16_t)(&w[0]),
		(*C.uint16_t)(&y[0]),
		C.int64_t(len(w)),
	)
}

func (*CTensor) FP16Div(x, w, y []uint16) {
	C.fp16_div(
		(*C.uint16_t)(&x[0]),
		(*C.uint16_t)(&w[0]),
		(*C.uint16_t)(&y[0]),
		C.int64_t(len(x)),
	)
}

func (*CTensor) FP16ScalarDiv(x uint16, w, y []uint16) {
	C.fp16_scalar_div(
		C.uint16_t(x),
		(*C.uint16_t)(&w[0]),
		(*C.uint16_t)(&y[0]),
		C.int64_t(len(w)),
	)
}

func (*CTensor) FP16Add(x, w, y []uint16) {
	C.fp16_add(
		(*C.uint16_t)(&x[0]),
		(*C.uint16_t)(&w[0]),
		(*C.uint16_t)(&y[0]),
		C.int64_t(len(x)),
	)
}

func (*CTensor) FP16ScalarAdd(x uint16, w, y []uint16) {
	C.fp16_scalar_add(
		C.uint16_t(x),
		(*C.uint16_t)(&w[0]),
		(*C.uint16_t)(&y[0]),
		C.int64_t(len(w)),
	)
}

func (*CTensor) FP16Sub(x, w, y []uint16) {
	C.fp16_sub(
		(*C.uint16_t)(&x[0]),
		(*C.uint16_t)(&w[0]),
		(*C.uint16_t)(&y[0]),
		C.int64_t(len(x)),
	)
}

func (*CTensor) FP16ScalarSub(x uint16, w, y []uint16) {
	C.fp16_scalar_sub(
		C.uint16_t(x),
		(*C.uint16_t)(&w[0]),
		(*C.uint16_t)(&y[0]),
		C.int64_t(len(w)),
	)
}

func (*CTensor) FP16Pow(x []uint16, n uint16, y []uint16) {
	C.fp16_pow(
		(*C.uint16_t)(&x[0]),
		C.uint16_t(n),
		(*C.uint16_t)(&y[0]),
		C.int64_t(len(x)),
	)
}
