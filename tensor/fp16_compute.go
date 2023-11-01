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
		return mulFP16(t, t2)
	case consts.Float32:
		return convert(t, consts.Float32).Mul(t2)
	default:
		panic(ErrNotSupported)
	}
}

func (t *Float16) MulScalar(n float32) gomath.Tensor {
	return scalarMulFP16(half.Encode(n), t)
}

func mulFP16(x, w gomath.Tensor) gomath.Tensor {
	return computeVectors(x, w, func(shapes []int64) gomath.Tensor {
		s := NewFloat16Storage(make([]uint16, sumShapes(shapes)))
		return NewFloat16WithStorage(s, shapes, gomath.WithDevice(x.Device()))
	}, scalarMulFP16, scalarMulFP16, func(ret, x, w any) {
		impl.FP16Mul(x.([]uint16), w.([]uint16), ret.([]uint16))
	})
}

func scalarMulFP16(scalar any, t2 gomath.Tensor) gomath.Tensor {
	s := scalar.(uint16)
	store := NewFloat16Storage(make([]uint16, t2.Storage().Size()))
	data := store.Data().([]uint16)
	parallel(int64(t2.Storage().Size()), int64(runtime.NumCPU()), func(offset, size int64, _ ...any) {
		goImpl.FP16ScalarMul(s, t2.Storage().Data().([]uint16)[offset:offset+size], data[offset:offset+size])
	})
	return NewFloat16WithStorage(store, t2.Size(),
		gomath.WithDevice(t2.Device()))
}

func (t *Float16) Div(t2 gomath.Tensor) gomath.Tensor {
	switch t2.Type() {
	case consts.Float16:
		return divFP16(t, t2)
	case consts.Float32:
		return convert(t, consts.Float32).Mul(t2)
	default:
		panic(ErrNotSupported)
	}
}

func (t *Float16) DivScalar(n float32) gomath.Tensor {
	return scalarMulFP16(half.Encode(1/n), t)
}

func divFP16(x, w gomath.Tensor) gomath.Tensor {
	return computeVectors(x, w, func(shapes []int64) gomath.Tensor {
		s := NewFloat16Storage(make([]uint16, sumShapes(shapes)))
		return NewFloat16WithStorage(s, shapes, gomath.WithDevice(x.Device()))
	}, scalarDivFP16, vectorDivFP16, func(ret, x, w any) {
		impl.FP16Div(x.([]uint16), w.([]uint16), ret.([]uint16))
	})
}

func scalarDivFP16(scalar any, t2 gomath.Tensor) gomath.Tensor {
	s := scalar.(uint16)
	store := NewFloat16Storage(make([]uint16, t2.Storage().Size()))
	data := store.Data().([]uint16)
	ptr := t2.Storage().Data().([]uint16)
	parallel(int64(t2.Storage().Size()), int64(runtime.NumCPU()), func(offset, size int64, _ ...any) {
		goImpl.FP16ScalarDiv(s, ptr[offset:offset+size], data[offset:offset+size])
	})
	return NewFloat16WithStorage(store, t2.Size(),
		gomath.WithDevice(t2.Device()))
}

func vectorDivFP16(scalar any, t2 gomath.Tensor) gomath.Tensor {
	s := half.Decode(scalar.(uint16))
	return scalarMulFP16(half.Encode(1/s), t2)
}

func (t *Float16) Add(t2 gomath.Tensor) gomath.Tensor {
	switch t2.Type() {
	case consts.Float16:
		return addFP16(t, t2)
	case consts.Float32:
		return convert(t, consts.Float32).Add(t2)
	default:
		panic(ErrNotSupported)
	}
}

func (t *Float16) AddScalar(n float32) gomath.Tensor {
	return scalarAddFP16(half.Encode(n), t)
}

func addFP16(x, w gomath.Tensor) gomath.Tensor {
	return computeVectors(x, w, func(shapes []int64) gomath.Tensor {
		s := NewFloat16Storage(make([]uint16, sumShapes(shapes)))
		return NewFloat16WithStorage(s, shapes, gomath.WithDevice(x.Device()))
	}, scalarAddFP16, scalarAddFP16, func(ret, x, w any) {
		impl.FP16Add(x.([]uint16), w.([]uint16), ret.([]uint16))
	})
}

func scalarAddFP16(scalar any, t2 gomath.Tensor) gomath.Tensor {
	s := scalar.(uint16)
	store := NewFloat16Storage(make([]uint16, t2.Storage().Size()))
	data := store.Data().([]uint16)
	parallel(int64(t2.Storage().Size()), int64(runtime.NumCPU()), func(offset, size int64, _ ...any) {
		goImpl.FP16ScalarAdd(s, t2.Storage().Data().([]uint16)[offset:offset+size], data[offset:offset+size])
	})
	return NewFloat16WithStorage(store, t2.Size(),
		gomath.WithDevice(t2.Device()))
}

func (t *Float16) Sub(t2 gomath.Tensor) gomath.Tensor {
	switch t2.Type() {
	case consts.Float16:
		return subFP16(t, t2)
	case consts.Float32:
		return convert(t, consts.Float32).Sub(t2)
	default:
		panic(ErrNotSupported)
	}
}

func (t *Float16) SubScalar(n float32) gomath.Tensor {
	return subScalarFP16(half.Encode(n), t)
}

func subFP16(x, w gomath.Tensor) gomath.Tensor {
	return computeVectors(x, w, func(shapes []int64) gomath.Tensor {
		s := NewFloat16Storage(make([]uint16, sumShapes(shapes)))
		return NewFloat16WithStorage(s, shapes, gomath.WithDevice(x.Device()))
	}, scalarSubFP16, subScalarFP16, func(ret, x, w any) {
		impl.FP16Sub(x.([]uint16), w.([]uint16), ret.([]uint16))
	})
}

func scalarSubFP16(scalar any, t2 gomath.Tensor) gomath.Tensor {
	s := scalar.(uint16)
	store := NewFloat16Storage(make([]uint16, t2.Storage().Size()))
	data := store.Data().([]uint16)
	parallel(int64(t2.Storage().Size()), int64(runtime.NumCPU()), func(offset, size int64, _ ...any) {
		goImpl.FP16ScalarSub(s, t2.Storage().Data().([]uint16)[offset:offset+size], data[offset:offset+size])
	})
	return NewFloat16WithStorage(store, t2.Size(),
		gomath.WithDevice(t2.Device()))
}

func subScalarFP16(scalar any, t2 gomath.Tensor) gomath.Tensor {
	s := scalar.(uint16)
	s = half.Encode(-half.Decode(s))
	return scalarAddFP16(s, t2)
}
