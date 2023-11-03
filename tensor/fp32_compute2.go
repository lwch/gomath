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
		return mulFP32(t, t2)
	default:
		panic(ErrNotSupported)
	}
}

func (t *Float32) MulScalar(n float32) gomath.Tensor {
	return scalarMulFP32(n, t)
}

func mulFP32(x, w gomath.Tensor) gomath.Tensor {
	return compute2(x, w, buildFP32,
		scalarMulFP32, scalarMulFP32, func(ret, x, w any) {
			impl.FP32Mul(x.([]float32), w.([]float32), ret.([]float32))
		})
}

func scalarMulFP32(scalar any, t2 gomath.Tensor) gomath.Tensor {
	s := scalar.(float32)
	store := NewFloat32Storage(make([]float32, t2.Storage().Size()))
	data := store.Data().([]float32)
	parallel(int64(t2.Storage().Size()), int64(runtime.NumCPU()), func(offset, size int64, _ ...any) {
		goImpl.FP32ScalarMul(s, t2.Storage().Data().([]float32)[offset:offset+size], data[offset:offset+size])
	})
	return NewFloat32WithStorage(store, t2.Size(),
		gomath.WithDevice(t2.Device()))
}

func (t *Float32) Div(t2 gomath.Tensor) gomath.Tensor {
	switch t2.Type() {
	case consts.Float16:
		return t.Div(convert(t2, consts.Float32))
	case consts.Float32:
		return divFP32(t, t2)
	default:
		panic(ErrNotSupported)
	}
}

func (t *Float32) DivScalar(n float32) gomath.Tensor {
	return scalarMulFP32(1/n, t)
}

func divFP32(x, w gomath.Tensor) gomath.Tensor {
	return compute2(x, w, buildFP32,
		scalarDivFP32, vectorDivFP32, func(ret, x, w any) {
			impl.FP32Div(x.([]float32), w.([]float32), ret.([]float32))
		})
}

func scalarDivFP32(scalar any, t2 gomath.Tensor) gomath.Tensor {
	s := scalar.(float32)
	store := NewFloat32Storage(make([]float32, t2.Storage().Size()))
	data := store.Data().([]float32)
	ptr := t2.Storage().Data().([]float32)
	parallel(int64(t2.Storage().Size()), int64(runtime.NumCPU()), func(offset, size int64, _ ...any) {
		goImpl.FP32ScalarDiv(s, ptr[offset:offset+size], data[offset:offset+size])
	})
	return NewFloat32WithStorage(store, t2.Size(),
		gomath.WithDevice(t2.Device()))
}

func vectorDivFP32(scalar any, t2 gomath.Tensor) gomath.Tensor {
	s := scalar.(float32)
	return scalarMulFP32(1/s, t2)
}

func (t *Float32) Add(t2 gomath.Tensor) gomath.Tensor {
	switch t2.Type() {
	case consts.Float16:
		return t.Add(convert(t2, consts.Float32))
	case consts.Float32:
		return addFP32(t, t2)
	default:
		panic(ErrNotSupported)
	}
}

func (t *Float32) AddScalar(n float32) gomath.Tensor {
	return scalarAddFP32(n, t)
}

func addFP32(x, w gomath.Tensor) gomath.Tensor {
	return compute2(x, w, buildFP32,
		scalarAddFP32, scalarAddFP32, func(ret, x, w any) {
			impl.FP32Add(x.([]float32), w.([]float32), ret.([]float32))
		})
}

func scalarAddFP32(scalar any, t2 gomath.Tensor) gomath.Tensor {
	s := scalar.(float32)
	store := NewFloat32Storage(make([]float32, t2.Storage().Size()))
	data := store.Data().([]float32)
	parallel(int64(t2.Storage().Size()), int64(runtime.NumCPU()), func(offset, size int64, _ ...any) {
		goImpl.FP32ScalarAdd(s, t2.Storage().Data().([]float32)[offset:offset+size], data[offset:offset+size])
	})
	return NewFloat32WithStorage(store, t2.Size(),
		gomath.WithDevice(t2.Device()))
}

func (t *Float32) Sub(t2 gomath.Tensor) gomath.Tensor {
	switch t2.Type() {
	case consts.Float16:
		return t.Sub(convert(t2, consts.Float32))
	case consts.Float32:
		return subFP32(t, t2)
	default:
		panic(ErrNotSupported)
	}
}

func (t *Float32) SubScalar(n float32) gomath.Tensor {
	return subScalarFP32(n, t)
}

func subFP32(x, w gomath.Tensor) gomath.Tensor {
	return compute2(x, w, buildFP32,
		scalarSubFP32, subScalarFP32, func(ret, x, w any) {
			impl.FP32Sub(x.([]float32), w.([]float32), ret.([]float32))
		})
}

func scalarSubFP32(scalar any, t2 gomath.Tensor) gomath.Tensor {
	s := scalar.(float32)
	store := NewFloat32Storage(make([]float32, t2.Storage().Size()))
	data := store.Data().([]float32)
	parallel(int64(t2.Storage().Size()), int64(runtime.NumCPU()), func(offset, size int64, _ ...any) {
		goImpl.FP32ScalarSub(s, t2.Storage().Data().([]float32)[offset:offset+size], data[offset:offset+size])
	})
	return NewFloat32WithStorage(store, t2.Size(),
		gomath.WithDevice(t2.Device()))
}

func subScalarFP32(scalar any, t2 gomath.Tensor) gomath.Tensor {
	s := scalar.(float32)
	return scalarAddFP32(-s, t2)
}
