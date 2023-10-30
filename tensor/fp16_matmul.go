package tensor

import (
	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
)

func (t *Float16) MatMul(t2 gomath.Tensor) gomath.Tensor {
	switch t2.Type() {
	case consts.Float16:
		return matMul(t, t2, func(shapes []int64) gomath.Tensor {
			data := make([]uint16, sumShapes(shapes))
			return NewFloat16WithStorage(
				NewFloat16Storage(data, shapes[len(shapes)-1]),
				shapes, gomath.WithDevice(t.Device()))
		}, t.matmul)
	case consts.Float32:
		return convert(t, consts.Float32).MatMul(t2)
	default:
		panic(ErrNotSupported)
	}
}

func (t *Float16) matmul(ret, t1, t2 gomath.Tensor, idx, row1, row2, _ int64) {
	vector1 := t1.Storage().Row(row1)
	vector2 := t2.Storage().Row(row2)
	ret.Storage().Set(idx, t.impl.FP16DotVector(
		vector1.([]uint16), vector2.([]uint16)))
}
