package tensor

import (
	"runtime"

	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
	"github.com/lwch/gomath/internal/half"
)

func (t *Float16) Mul(t2 gomath.Tensor) gomath.Tensor {
	switch t2.(type) {
	case *Float16:
		return computeVectors(t, t2, func(shapes []int64) gomath.Tensor {
			s := NewFloat16Storage(make([]uint16, sumShapes(shapes)))
			return NewFloat16WithStorage(s, shapes, gomath.WithDevice(t.Device()))
		}, t.scalarMul, t.scalarMul, t.mul)
	case *Float32:
		return convert(t, consts.Float32).Mul(t2)
	default:
		panic(ErrNotSupported)
	}
}

func (t *Float16) MulScalar(n float32) gomath.Tensor {
	return t.scalarMul(half.Encode(n), t)
}

func (t *Float16) scalarMul(scalar any, t2 gomath.Tensor) gomath.Tensor {
	s := scalar.(uint16)
	store := NewFloat16Storage(make([]uint16, t2.Storage().Size()))
	data := store.Data().([]uint16)
	parallel(int64(t2.Storage().Size()), int64(runtime.NumCPU()), func(offset, size int64, _ ...any) {
		goImpl.FP16ScalarMul(s, t2.Storage().Data().([]uint16)[offset:offset+size], data[offset:offset+size])
	})
	return NewFloat16WithStorage(store, t2.Size(),
		gomath.WithDevice(t.Device()))
}

func (t *Float16) mul(ret, dx, dw any) {
	impl.FP16Mul(dx.([]uint16), dw.([]uint16), ret.([]uint16))
}

func (t *Float16) Div(t2 gomath.Tensor) gomath.Tensor {
	switch t2.(type) {
	case *Float16:
		return computeVectors(t, t2, func(shapes []int64) gomath.Tensor {
			s := NewFloat16Storage(make([]uint16, sumShapes(shapes)))
			return NewFloat16WithStorage(s, shapes, gomath.WithDevice(t.Device()))
		}, t.scalarDiv, t.vectorDiv, t.div)
	case *Float32:
		return convert(t, consts.Float32).Mul(t2)
	default:
		panic(ErrNotSupported)
	}
}

func (t *Float16) DivScalar(n float32) gomath.Tensor {
	return t.scalarMul(half.Encode(1/n), t)
}

func (t *Float16) scalarDiv(scalar any, t2 gomath.Tensor) gomath.Tensor {
	s := scalar.(uint16)
	store := NewFloat16Storage(make([]uint16, t2.Storage().Size()))
	data := store.Data().([]uint16)
	ptr := t2.Storage().Data().([]uint16)
	parallel(int64(t2.Storage().Size()), int64(runtime.NumCPU()), func(offset, size int64, _ ...any) {
		goImpl.FP16ScalarDiv(s, ptr[offset:offset+size], data[offset:offset+size])
	})
	return NewFloat16WithStorage(store, t2.Size(),
		gomath.WithDevice(t.Device()))
}

func (t *Float16) vectorDiv(scalar any, t2 gomath.Tensor) gomath.Tensor {
	s := half.Decode(scalar.(uint16))
	return t.scalarMul(half.Encode(1/s), t2)
}

func (t *Float16) div(ret, dx, dw any) {
	impl.FP16Div(dx.([]uint16), dw.([]uint16), ret.([]uint16))
}

func (t *Float16) Add(t2 gomath.Tensor) gomath.Tensor {
	switch t2.(type) {
	case *Float16:
		return computeVectors(t, t2, func(shapes []int64) gomath.Tensor {
			s := NewFloat16Storage(make([]uint16, sumShapes(shapes)))
			return NewFloat16WithStorage(s, shapes, gomath.WithDevice(t.Device()))
		}, t.scalarAdd, t.scalarAdd, t.add)
	case *Float32:
		return convert(t, consts.Float32).Add(t2)
	default:
		panic(ErrNotSupported)
	}
}

func (t *Float16) AddScalar(n float32) gomath.Tensor {
	return t.scalarAdd(half.Encode(n), t)
}

func (t *Float16) scalarAdd(scalar any, t2 gomath.Tensor) gomath.Tensor {
	s := scalar.(uint16)
	store := NewFloat16Storage(make([]uint16, t2.Storage().Size()))
	data := store.Data().([]uint16)
	parallel(int64(t2.Storage().Size()), int64(runtime.NumCPU()), func(offset, size int64, _ ...any) {
		goImpl.FP16ScalarAdd(s, t2.Storage().Data().([]uint16)[offset:offset+size], data[offset:offset+size])
	})
	return NewFloat16WithStorage(store, t2.Size(),
		gomath.WithDevice(t.Device()))
}

func (t *Float16) add(ret, dx, dw any) {
	impl.FP16Add(dx.([]uint16), dw.([]uint16), ret.([]uint16))
}

func (t *Float16) Sub(t2 gomath.Tensor) gomath.Tensor {
	switch t2.(type) {
	case *Float16:
		return computeVectors(t, t2, func(shapes []int64) gomath.Tensor {
			s := NewFloat16Storage(make([]uint16, sumShapes(shapes)))
			return NewFloat16WithStorage(s, shapes, gomath.WithDevice(t.Device()))
		}, t.scalarSub, t.subScalar, t.sub)
	case *Float32:
		return convert(t, consts.Float32).Add(t2)
	default:
		panic(ErrNotSupported)
	}
}

func (t *Float16) SubScalar(n float32) gomath.Tensor {
	return t.subScalar(half.Encode(n), t)
}

func (t *Float16) scalarSub(scalar any, t2 gomath.Tensor) gomath.Tensor {
	s := scalar.(uint16)
	store := NewFloat16Storage(make([]uint16, t2.Storage().Size()))
	data := store.Data().([]uint16)
	parallel(int64(t2.Storage().Size()), int64(runtime.NumCPU()), func(offset, size int64, _ ...any) {
		goImpl.FP16ScalarSub(s, t2.Storage().Data().([]uint16)[offset:offset+size], data[offset:offset+size])
	})
	return NewFloat16WithStorage(store, t2.Size(),
		gomath.WithDevice(t.Device()))
}

func (t *Float16) subScalar(scalar any, t2 gomath.Tensor) gomath.Tensor {
	s := scalar.(uint16)
	s = half.Encode(-half.Decode(s))
	return t.scalarAdd(s, t2)
}

func (t *Float16) sub(ret, dx, dw any) {
	impl.FP16Sub(dx.([]uint16), dw.([]uint16), ret.([]uint16))
}
