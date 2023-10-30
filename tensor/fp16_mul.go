package tensor

import (
	"runtime"

	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
	"github.com/lwch/gomath/internal/half"
)

func (t *Float16) Mul(t2 gomath.Tensor) gomath.Tensor {
	switch t2.Type() {
	case consts.Float16:
		return computeVectors(t, t2, func(shapes []int64) gomath.Tensor {
			return NewFloat16(nil, shapes,
				gomath.WithDevice(t.Device()))
		}, t.mulScalar, t.mulScalar, t.mulVector)
	case consts.Float32:
		return convert(t, consts.Float32).Mul(t2)
	default:
		panic(ErrNotSupported)
	}
}

func (t *Float16) mulScalar(scalar any, t2 gomath.Tensor) gomath.Tensor {
	s := scalar.(uint16)
	data := make([]uint16, t2.Storage().Size())
	parallel(int64(t2.Storage().Size()), int64(runtime.NumCPU()), func(offset, size int64, _ ...int64) {
		t.impl.FP16MulScalar(t2.Storage().Data().([]uint16)[offset:offset+size], s, data[offset:offset+size])
	})
	return NewFloat16Raw(data, t.Size(),
		gomath.WithDevice(t.Device()))
}

func (t *Float16) mulVector(ret, t1, t2 gomath.Tensor, idx, row1, row2 int64) {
	t.impl.FP16MulVector(
		t1.Storage().Row(row1).([]uint16),
		t2.Storage().Row(row2).([]uint16),
		ret.Storage().Row(idx).([]uint16))
}

func (t *Float16) Div(t2 gomath.Tensor) gomath.Tensor {
	switch t2.Type() {
	case consts.Float16:
		return computeVectors(t, t2, func(shapes []int64) gomath.Tensor {
			return NewFloat16(nil, shapes,
				gomath.WithDevice(t.Device()))
		}, t.scalarDivVector, t.vectorDivScalar, t.divVector)
	case consts.Float32:
		return convert(t, consts.Float32).Mul(t2)
	default:
		panic(ErrNotSupported)
	}
}

func (t *Float16) scalarDivVector(scalar any, t2 gomath.Tensor) gomath.Tensor {
	s := scalar.(uint16)
	data := make([]uint16, t2.Storage().Size())
	ptr := t2.Storage().Data().([]uint16)
	parallel(int64(t2.Storage().Size()), int64(runtime.NumCPU()), func(offset, size int64, _ ...int64) {
		t.impl.FP16ScalarDivVector(s, ptr[offset:offset+size], data[offset:offset+size])
	})
	return NewFloat16Raw(data, t.Size(),
		gomath.WithDevice(t.Device()))
}

func (t *Float16) vectorDivScalar(scalar any, t2 gomath.Tensor) gomath.Tensor {
	s := half.Decode(scalar.(uint16))
	return t.mulScalar(half.Encode(1/s), t2)
}

func (t *Float16) divVector(ret, t1, t2 gomath.Tensor, idx, row1, row2 int64) {
	t.impl.FP16DivVector(
		t1.Storage().Row(row1).([]uint16),
		t2.Storage().Row(row2).([]uint16),
		ret.Storage().Row(idx).([]uint16))
}
