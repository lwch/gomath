package tensor

import (
	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
)

func (t *Float32) MatMul(t2 gomath.Tensor) gomath.Tensor {
	switch t2.Type() {
	case consts.Float16:
		return t.MatMul(convert(t2, consts.Float32).(*Float32))
	case consts.Float32:
		return matMul(t, t2,
			func(shapes []int64) gomath.Tensor {
				data := make([]float32, sumShapes(shapes))
				return NewFloat32WithStorage(
					NewFloat32Storage(data, shapes[len(shapes)-1]),
					shapes, gomath.WithDevice(t.Device()))
			}, t.matmul)
	default:
		panic("not supported")
	}
}

func (t *Float32) matmul(ret, dx, dw any, col int64) {
	ret.([]float32)[col] = impl.FP32Dot(dx.([]float32), dw.([]float32))
}
