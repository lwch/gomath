package tensor

import "github.com/lwch/gomath"

func (t *Float32) Pow(n float32) gomath.Tensor {
	return powFP32(n, t)
}

func powFP32(n any, t gomath.Tensor) gomath.Tensor {
	return compute11(t, buildFP32, func(ret, x any) {
		impl.FP32Pow(x.([]float32), n.(float32), ret.([]float32))
	})
}
