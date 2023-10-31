package tensor

import (
	"runtime"

	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
	"github.com/lwch/gomath/internal/tensor"
)

func (t *Float32) Mul(t2 gomath.Tensor) gomath.Tensor {
	switch t2.Type() {
	case consts.Float16:
		return t.Mul(convert(t2, consts.Float32))
	case consts.Float32:
		return computeVectors(t, t2, func(shapes []int64) gomath.Tensor {
			s := NewFloat32Storage(make([]float32, sumShapes(shapes)), shapes[len(shapes)-1])
			return NewFloat32WithStorage(s, shapes, gomath.WithDevice(t.Device()))
		}, t.mulScalar, t.mulScalar, t.mulVector)
	default:
		panic(ErrNotSupported)
	}
}

func (t *Float32) mulScalar(impl tensor.TensorImpl, scalar any, t2 gomath.Tensor, d int64) gomath.Tensor {
	s := scalar.(float32)
	store := NewFloat32Storage(make([]float32, t2.Storage().Size()), d)
	data := store.Data().([]float32)
	parallel(int64(t2.Storage().Size()), int64(runtime.NumCPU()), func(offset, size int64, _ ...any) {
		impl.FP32MulScalar(s, t2.Storage().Data().([]float32)[offset:offset+size], data[offset:offset+size])
	})
	return NewFloat32WithStorage(store, t.Size(),
		gomath.WithDevice(t.Device()))
}

func (t *Float32) mulVector(ret, dx, dw any) {
	impl.FP32MulVector(dx.([]float32), dw.([]float32), ret.([]float32))
}

func (t *Float32) Div(t2 gomath.Tensor) gomath.Tensor {
	switch t2.Type() {
	case consts.Float16:
		return t.Mul(convert(t2, consts.Float32))
	case consts.Float32:
		return computeVectors(t, t2, func(shapes []int64) gomath.Tensor {
			s := NewFloat32Storage(make([]float32, sumShapes(shapes)), shapes[len(shapes)-1])
			return NewFloat32WithStorage(s, shapes, gomath.WithDevice(t.Device()))
		}, t.scalarDivVector, t.vectorDivScalar, t.divVector)
	default:
		panic(ErrNotSupported)
	}
}

func (t *Float32) scalarDivVector(impl tensor.TensorImpl, scalar any, t2 gomath.Tensor, d int64) gomath.Tensor {
	s := scalar.(float32)
	store := NewFloat32Storage(make([]float32, t2.Storage().Size()), d)
	data := store.Data().([]float32)
	ptr := t2.Storage().Data().([]float32)
	parallel(int64(t2.Storage().Size()), int64(runtime.NumCPU()), func(offset, size int64, _ ...any) {
		impl.FP32DivScalar(s, ptr[offset:offset+size], data[offset:offset+size])
	})
	return NewFloat32WithStorage(store, t.Size(),
		gomath.WithDevice(t.Device()))
}

func (t *Float32) vectorDivScalar(impl tensor.TensorImpl, scalar any, t2 gomath.Tensor, d int64) gomath.Tensor {
	s := scalar.(float32)
	return t.mulScalar(impl, 1/s, t2, d)
}

func (t *Float32) divVector(ret, dx, dw any) {
	impl.FP32DivVector(dx.([]float32), dw.([]float32), ret.([]float32))
}

func (t *Float32) Add(t2 gomath.Tensor) gomath.Tensor {
	switch t2.Type() {
	case consts.Float16:
		return t.Add(convert(t2, consts.Float32))
	case consts.Float32:
		return computeVectors(t, t2, func(shapes []int64) gomath.Tensor {
			s := NewFloat32Storage(make([]float32, sumShapes(shapes)), shapes[len(shapes)-1])
			return NewFloat32WithStorage(s, shapes, gomath.WithDevice(t.Device()))
		}, t.addScalar, t.addScalar, t.addVector)
	default:
		panic(ErrNotSupported)
	}
}

func (t *Float32) addScalar(impl tensor.TensorImpl, scalar any, t2 gomath.Tensor, d int64) gomath.Tensor {
	s := scalar.(float32)
	store := NewFloat32Storage(make([]float32, t2.Storage().Size()), d)
	data := store.Data().([]float32)
	parallel(int64(t2.Storage().Size()), int64(runtime.NumCPU()), func(offset, size int64, _ ...any) {
		impl.FP32AddScalar(s, t2.Storage().Data().([]float32)[offset:offset+size], data[offset:offset+size])
	})
	return NewFloat32WithStorage(store, t.Size(),
		gomath.WithDevice(t.Device()))
}

func (t *Float32) addVector(ret, dx, dw any) {
	impl.FP32AddVector(dx.([]float32), dw.([]float32), ret.([]float32))
}
