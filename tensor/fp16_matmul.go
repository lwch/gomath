package tensor

import (
	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
)

func (t *Float16) MatMul(t2 gomath.Tensor) gomath.Tensor {
	switch t2.(type) {
	case *Float16:
		return matMul(t, t2, func(shapes []int64) gomath.Tensor {
			data := make([]uint16, sumShapes(shapes))
			return NewFloat16WithStorage(
				NewFloat16Storage(data, shapes[len(shapes)-1]),
				shapes, gomath.WithDevice(t.Device()))
		}, t.matmul)
	case *Float32:
		return convert(t, consts.Float32).MatMul(t2)
	default:
		panic(ErrNotSupported)
	}
}

func (t *Float16) matmul(ret, dx, dw any, col int64) {
	ret.([]uint16)[col] = impl.FP16Dot(dx.([]uint16), dw.([]uint16))
}
