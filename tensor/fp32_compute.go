package tensor

import (
	"runtime"

	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
)

func (t *Float32) Mul(t2 gomath.Tensor) gomath.Tensor {
	switch t2.(type) {
	case *Float16:
		return t.Mul(convert(t2, consts.Float32))
	case *Float32:
		return computeVectors(t, t2, func(shapes []int64) gomath.Tensor {
			s := NewFloat32Storage(make([]float32, sumShapes(shapes)), shapes[len(shapes)-1])
			return NewFloat32WithStorage(s, shapes, gomath.WithDevice(t.Device()))
		}, t.scalarMul, t.scalarMul, t.mul)
	default:
		panic(ErrNotSupported)
	}
}

func (t *Float32) MulScalar(n float32) gomath.Tensor {
	return t.scalarMul(n, t, t.Size()[t.Dim()-1])
}

func (t *Float32) scalarMul(scalar any, t2 gomath.Tensor, d int64) gomath.Tensor {
	s := scalar.(float32)
	store := NewFloat32Storage(make([]float32, t2.Storage().Size()), d)
	data := store.Data().([]float32)
	parallel(int64(t2.Storage().Size()), int64(runtime.NumCPU()), func(offset, size int64, _ ...any) {
		goImpl.FP32ScalarMul(s, t2.Storage().Data().([]float32)[offset:offset+size], data[offset:offset+size])
	})
	return NewFloat32WithStorage(store, t2.Size(),
		gomath.WithDevice(t.Device()))
}

func (t *Float32) mul(ret, dx, dw any) {
	impl.FP32Mul(dx.([]float32), dw.([]float32), ret.([]float32))
}

func (t *Float32) Div(t2 gomath.Tensor) gomath.Tensor {
	switch t2.(type) {
	case *Float16:
		return t.Mul(convert(t2, consts.Float32))
	case *Float32:
		return computeVectors(t, t2, func(shapes []int64) gomath.Tensor {
			s := NewFloat32Storage(make([]float32, sumShapes(shapes)), shapes[len(shapes)-1])
			return NewFloat32WithStorage(s, shapes, gomath.WithDevice(t.Device()))
		}, t.scalarDiv, t.vectorDiv, t.div)
	default:
		panic(ErrNotSupported)
	}
}

func (t *Float32) DivScalar(n float32) gomath.Tensor {
	return t.scalarMul(1/n, t, t.Size()[t.Dim()-1])
}

func (t *Float32) scalarDiv(scalar any, t2 gomath.Tensor, d int64) gomath.Tensor {
	s := scalar.(float32)
	store := NewFloat32Storage(make([]float32, t2.Storage().Size()), d)
	data := store.Data().([]float32)
	ptr := t2.Storage().Data().([]float32)
	parallel(int64(t2.Storage().Size()), int64(runtime.NumCPU()), func(offset, size int64, _ ...any) {
		goImpl.FP32ScalarDiv(s, ptr[offset:offset+size], data[offset:offset+size])
	})
	return NewFloat32WithStorage(store, t2.Size(),
		gomath.WithDevice(t.Device()))
}

func (t *Float32) vectorDiv(scalar any, t2 gomath.Tensor, d int64) gomath.Tensor {
	s := scalar.(float32)
	return t.scalarMul(1/s, t2, d)
}

func (t *Float32) div(ret, dx, dw any) {
	impl.FP32Div(dx.([]float32), dw.([]float32), ret.([]float32))
}

func (t *Float32) Add(t2 gomath.Tensor) gomath.Tensor {
	switch t2.(type) {
	case *Float16:
		return t.Add(convert(t2, consts.Float32))
	case *Float32:
		return computeVectors(t, t2, func(shapes []int64) gomath.Tensor {
			s := NewFloat32Storage(make([]float32, sumShapes(shapes)), shapes[len(shapes)-1])
			return NewFloat32WithStorage(s, shapes, gomath.WithDevice(t.Device()))
		}, t.scalarAdd, t.scalarAdd, t.add)
	default:
		panic(ErrNotSupported)
	}
}

func (t *Float32) AddScalar(n float32) gomath.Tensor {
	return t.scalarAdd(n, t, t.Size()[t.Dim()-1])
}

func (t *Float32) scalarAdd(scalar any, t2 gomath.Tensor, d int64) gomath.Tensor {
	s := scalar.(float32)
	store := NewFloat32Storage(make([]float32, t2.Storage().Size()), d)
	data := store.Data().([]float32)
	parallel(int64(t2.Storage().Size()), int64(runtime.NumCPU()), func(offset, size int64, _ ...any) {
		goImpl.FP32ScalarAdd(s, t2.Storage().Data().([]float32)[offset:offset+size], data[offset:offset+size])
	})
	return NewFloat32WithStorage(store, t2.Size(),
		gomath.WithDevice(t.Device()))
}

func (t *Float32) add(ret, dx, dw any) {
	impl.FP32Add(dx.([]float32), dw.([]float32), ret.([]float32))
}
