package ctensor

// #include "fp32/fp32.h"
import "C"

func (*CTensor) FP32DotVector(x, w []float32) float32 {
	return float32(C.fp32_dot_vector(
		(*C.float)(&x[0]),
		(*C.float)(&w[0]),
		C.int64_t(len(x)),
	))
}

func (*CTensor) FP32MulVector(x, w, y []float32) {
	C.fp32_mul_vector(
		(*C.float)(&x[0]),
		(*C.float)(&w[0]),
		(*C.float)(&y[0]),
		C.int64_t(len(x)),
	)
}

func (*CTensor) FP32MulScalar(x []float32, w float32, y []float32) {
	C.fp32_mul_scalar(
		(*C.float)(&x[0]),
		C.float(w),
		(*C.float)(&y[0]),
		C.int64_t(len(x)),
	)
}

func (*CTensor) FP32DivVector(x, w, y []float32) {
	C.fp32_div_vector(
		(*C.float)(&x[0]),
		(*C.float)(&w[0]),
		(*C.float)(&y[0]),
		C.int64_t(len(x)),
	)
}
