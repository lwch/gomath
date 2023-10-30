package tensor

import (
	"runtime"

	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
)

func (t *Float32) Mul(t2 gomath.Tensor) gomath.Tensor {
	switch t2.Type() {
	case consts.Float16:
		return t.Mul(convert(t2, consts.Float32))
	case consts.Float32:
		return computeVectors(t, t2, func(shapes []int64) gomath.Tensor {
			return NewFloat32(nil, shapes,
				gomath.WithDevice(t.Device()))
		}, t.mulScalar, t.mulScalar, t.mulVector)
	default:
		panic(ErrNotSupported)
	}
}

func (t *Float32) mulScalar(scalar any, t2 gomath.Tensor) gomath.Tensor {
	s := scalar.(float32)
	data := make([]float32, t2.Storage().Size())
	parallel(int64(t2.Storage().Size()), int64(runtime.NumCPU()), func(offset, size int64, _ ...int64) {
		t.impl.FP32MulScalar(t2.Storage().Data().([]float32)[offset:offset+size], s, data[offset:offset+size])
	})
	return NewFloat32(data, t.Size(),
		gomath.WithDevice(t.Device()))
}

func (t *Float32) mulVector(ret, t1, t2 gomath.Tensor, idx, row1, row2 int64) {
	t.impl.FP32MulVector(
		t1.Storage().Row(row1).([]float32),
		t2.Storage().Row(row2).([]float32),
		ret.Storage().Row(idx).([]float32))
}

func (t *Float32) Div(t2 gomath.Tensor) gomath.Tensor {
	switch t2.Type() {
	case consts.Float16:
		return t.Div(convert(t2, consts.Float32))
	case consts.Float32:
		return computeVectors(t, t2, func(shapes []int64) gomath.Tensor {
			return NewFloat32(nil, shapes,
				gomath.WithDevice(t.Device()))
		}, t.scalarDivVector, t.vectorDivScalar, t.divVector)
	default:
		panic(ErrNotSupported)
	}
}

func (t *Float32) scalarDivVector(scalar any, t2 gomath.Tensor) gomath.Tensor {
	s := scalar.(float32)
	data := make([]float32, t2.Storage().Size())
	ptr := t2.Storage().Data().([]float32)
	parallel(int64(t2.Storage().Size()), int64(runtime.NumCPU()), func(offset, size int64, _ ...int64) {
		t.impl.FP32ScalarDivVector(s, ptr[offset:offset+size], data[offset:offset+size])
	})
	return NewFloat32(data, t.Size(),
		gomath.WithDevice(t.Device()))
}

func (t *Float32) vectorDivScalar(scalar any, t2 gomath.Tensor) gomath.Tensor {
	s := scalar.(float32)
	return t.mulScalar(1/s, t2)
}

func (t *Float32) divVector(ret, t1, t2 gomath.Tensor, idx, row1, row2 int64) {
	t.impl.FP32DivVector(
		t1.Storage().Row(row1).([]float32),
		t2.Storage().Row(row2).([]float32),
		ret.Storage().Row(idx).([]float32))
}
