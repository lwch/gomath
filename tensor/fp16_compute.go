package tensor

import (
	"runtime"

	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
	"github.com/lwch/gomath/internal/half"
	"github.com/lwch/gomath/internal/tensor"
)

func (t *Float16) Mul(t2 gomath.Tensor) gomath.Tensor {
	switch t2.Type() {
	case consts.Float16:
		return computeVectors(t, t2, func(shapes []int64) gomath.Tensor {
			s := NewFloat16Storage(make([]uint16, sumShapes(shapes)), shapes[len(shapes)-1])
			return NewFloat16WithStorage(s, shapes, gomath.WithDevice(t.Device()))
		}, t.mulScalar, t.mulScalar, t.mulVector)
	case consts.Float32:
		return convert(t, consts.Float32).Mul(t2)
	default:
		panic(ErrNotSupported)
	}
}

func (t *Float16) mulScalar(impl tensor.TensorImpl, scalar any, t2 gomath.Tensor, d int64) gomath.Tensor {
	s := scalar.(uint16)
	store := NewFloat16Storage(make([]uint16, t2.Storage().Size()), d)
	data := store.Data().([]uint16)
	parallel(int64(t2.Storage().Size()), int64(runtime.NumCPU()), func(offset, size int64, _ ...any) {
		impl.FP16MulScalar(s, t2.Storage().Data().([]uint16)[offset:offset+size], data[offset:offset+size])
	})
	return NewFloat16WithStorage(store, t.Size(),
		gomath.WithDevice(t.Device()))
}

func (t *Float16) mulVector(ret, dx, dw any) {
	impl.FP16MulVector(dx.([]uint16), dw.([]uint16), ret.([]uint16))
}

func (t *Float16) Div(t2 gomath.Tensor) gomath.Tensor {
	switch t2.Type() {
	case consts.Float16:
		return computeVectors(t, t2, func(shapes []int64) gomath.Tensor {
			s := NewFloat16Storage(make([]uint16, sumShapes(shapes)), shapes[len(shapes)-1])
			return NewFloat16WithStorage(s, shapes, gomath.WithDevice(t.Device()))
		}, t.scalarDivVector, t.vectorDivScalar, t.divVector)
	case consts.Float32:
		return convert(t, consts.Float32).Mul(t2)
	default:
		panic(ErrNotSupported)
	}
}

func (t *Float16) scalarDivVector(impl tensor.TensorImpl, scalar any, t2 gomath.Tensor, d int64) gomath.Tensor {
	s := scalar.(uint16)
	store := NewFloat16Storage(make([]uint16, t2.Storage().Size()), d)
	data := store.Data().([]uint16)
	ptr := t2.Storage().Data().([]uint16)
	parallel(int64(t2.Storage().Size()), int64(runtime.NumCPU()), func(offset, size int64, _ ...any) {
		impl.FP16DivScalar(s, ptr[offset:offset+size], data[offset:offset+size])
	})
	return NewFloat16WithStorage(store, t.Size(),
		gomath.WithDevice(t.Device()))
}

func (t *Float16) vectorDivScalar(impl tensor.TensorImpl, scalar any, t2 gomath.Tensor, d int64) gomath.Tensor {
	s := half.Decode(scalar.(uint16))
	return t.mulScalar(impl, half.Encode(1/s), t2, d)
}

func (t *Float16) divVector(ret, dx, dw any) {
	impl.FP16DivVector(dx.([]uint16), dw.([]uint16), ret.([]uint16))
}

func (t *Float16) Add(t2 gomath.Tensor) gomath.Tensor {
	switch t2.Type() {
	case consts.Float16:
		return computeVectors(t, t2, func(shapes []int64) gomath.Tensor {
			s := NewFloat16Storage(make([]uint16, sumShapes(shapes)), shapes[len(shapes)-1])
			return NewFloat16WithStorage(s, shapes, gomath.WithDevice(t.Device()))
		}, t.addScalar, t.addScalar, t.addVector)
	case consts.Float32:
		return convert(t, consts.Float32).Add(t2)
	default:
		panic(ErrNotSupported)
	}
}

func (t *Float16) addScalar(impl tensor.TensorImpl, scalar any, t2 gomath.Tensor, d int64) gomath.Tensor {
	s := scalar.(uint16)
	store := NewFloat16Storage(make([]uint16, t2.Storage().Size()), d)
	data := store.Data().([]uint16)
	parallel(int64(t2.Storage().Size()), int64(runtime.NumCPU()), func(offset, size int64, _ ...any) {
		impl.FP16AddScalar(s, t2.Storage().Data().([]uint16)[offset:offset+size], data[offset:offset+size])
	})
	return NewFloat16WithStorage(store, t.Size(),
		gomath.WithDevice(t.Device()))
}

func (t *Float16) addVector(ret, dx, dw any) {
	impl.FP16AddVector(dx.([]uint16), dw.([]uint16), ret.([]uint16))
}
