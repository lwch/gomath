package tensor

import (
	"github.com/lwch/gomath"
	"github.com/lwch/gomath/internal/half"
)

func (t *Float16) Pow(n float32) gomath.Tensor {
	return powFP16(n, t)
}

func powFP16(n any, t gomath.Tensor) gomath.Tensor {
	return compute11(t, buildFP16, func(ret, x any) {
		impl.FP16Pow(x.([]uint16), half.Encode(n.(float32)), ret.([]uint16))
	})
}
