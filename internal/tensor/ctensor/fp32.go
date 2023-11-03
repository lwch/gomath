package ctensor

// #include "fp32/fp32.h"
import "C"

func (*CTensor) FP32Dot(x, w []float32) float32 {
	return float32(C.fp32_dot(
		(*C.float)(&x[0]),
		(*C.float)(&w[0]),
		C.int64_t(len(x)),
	))
}

func (*CTensor) FP32Mul(x, w, y []float32) {
	C.fp32_mul(
		(*C.float)(&x[0]),
		(*C.float)(&w[0]),
		(*C.float)(&y[0]),
		C.int64_t(len(x)),
	)
}

func (*CTensor) FP32ScalarMul(x float32, w, y []float32) {
	C.fp32_scalar_mul(
		C.float(x),
		(*C.float)(&w[0]),
		(*C.float)(&y[0]),
		C.int64_t(len(w)),
	)
}

func (*CTensor) FP32Div(x, w, y []float32) {
	C.fp32_div(
		(*C.float)(&x[0]),
		(*C.float)(&w[0]),
		(*C.float)(&y[0]),
		C.int64_t(len(x)),
	)
}

func (*CTensor) FP32ScalarDiv(x float32, w, y []float32) {
	C.fp32_scalar_div(
		C.float(x),
		(*C.float)(&w[0]),
		(*C.float)(&y[0]),
		C.int64_t(len(w)),
	)
}

func (*CTensor) FP32Add(x, w, y []float32) {
	C.fp32_add(
		(*C.float)(&x[0]),
		(*C.float)(&w[0]),
		(*C.float)(&y[0]),
		C.int64_t(len(x)),
	)
}

func (*CTensor) FP32ScalarAdd(x float32, w, y []float32) {
	C.fp32_scalar_add(
		C.float(x),
		(*C.float)(&w[0]),
		(*C.float)(&y[0]),
		C.int64_t(len(w)),
	)
}

func (*CTensor) FP32Sub(x, w, y []float32) {
	C.fp32_sub(
		(*C.float)(&x[0]),
		(*C.float)(&w[0]),
		(*C.float)(&y[0]),
		C.int64_t(len(x)),
	)
}

func (*CTensor) FP32ScalarSub(x float32, w, y []float32) {
	C.fp32_scalar_sub(
		C.float(x),
		(*C.float)(&w[0]),
		(*C.float)(&y[0]),
		C.int64_t(len(w)),
	)
}

func (*CTensor) FP32Pow(x []float32, n float32, y []float32) {
	C.fp32_pow(
		(*C.float)(&x[0]),
		C.float(n),
		(*C.float)(&y[0]),
		C.int64_t(len(x)),
	)
}
