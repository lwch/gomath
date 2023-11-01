package tensor

import (
	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
)

func (t *Float32) MatMul(t2 gomath.Tensor) gomath.Tensor {
	switch t2.(type) {
	case *Float16:
		return t.MatMul(convert(t2, consts.Float32).(*Float32))
	case *Float32:
		return matMulFP32(t, t2)
	case *View:
		if t2.Type() == consts.Float32 {
			return matMulFP32(t, t2)
		} else {
			return t.MatMul(convert(t2, consts.Float32).(*Float32))
		}
	default:
		panic("not supported")
	}
}

func matMulFP32(x, w gomath.Tensor) gomath.Tensor {
	return matMul(x, w,
		func(shapes []int64) gomath.Tensor {
			data := make([]float32, sumShapes(shapes))
			return NewFloat32WithStorage(
				NewFloat32Storage(data),
				shapes, gomath.WithDevice(x.Device()))
		}, func(ret, x, w any, col int64) {
			ret.([]float32)[col] = impl.FP32Dot(x.([]float32), w.([]float32))
		})
}
