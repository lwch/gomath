package tensor

import (
	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
)

func (t *Float16) MatMul(t2 gomath.Tensor) gomath.Tensor {
	switch t2.(type) {
	case *Float16:
		return matMulFP16(t, t2)
	case *Float32:
		return convert(t, consts.Float32).MatMul(t2)
	case *View:
		if t2.Type() == consts.Float16 {
			return matMulFP16(t, t2)
		} else {
			return convert(t, consts.Float32).MatMul(t2)
		}
	default:
		panic(ErrNotSupported)
	}
}

func matMulFP16(x, w gomath.Tensor) gomath.Tensor {
	return matMul(x, w, func(shapes []int64) gomath.Tensor {
		data := make([]uint16, sumShapes(shapes))
		return NewFloat16WithStorage(
			NewFloat16Storage(data),
			shapes, gomath.WithDevice(x.Device()))
	}, func(ret, x, w any, col int64) {
		ret.([]uint16)[col] = impl.FP16Dot(x.([]uint16), w.([]uint16))
	})
}
