package tensor

import (
	"fmt"
	"runtime"

	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
)

func (t *Float32) Mul(t2 gomath.Tensor) gomath.Tensor {
	switch t2.Type() {
	case consts.Float16:
		return t.mul(convert(t2, consts.Float32).(*Float32))
	case consts.Float32:
		return t.mul(t2.(*Float32))
	default:
		panic(ErrNotSupported)
	}
}

func (t *Float32) mul(t2 *Float32) gomath.Tensor {
	size1, d1 := splitSize1(t)
	size2, d2 := splitSize1(t2)
	if len(size1) == 0 {
		if d1 == 1 {
			// scalar * t2
			return t.mulScalar(t2, t.data[0])
		} else {
			// vector * t2
			return t.mulVector(t2, t.data, count(size2))
		}
	}
	if len(size2) == 0 {
		if d2 == 1 {
			// t * scalar
			return t.mulScalar(t, t2.data[0])
		} else {
			// t * vector
			return t.mulVector(t, t2.data, count(size1))
		}
	}
	if d1 != d2 {
		panic(fmt.Errorf("dimension mismatch: %v and %v", t.Size(), t2.Size()))
	}
	if !sizeMatch(size1, size2) {
		panic(ErrBroadcast)
	}
	head := count(size1)
	data := make([]float32, head*d1)
	parallel(head*d1, int64(runtime.NumCPU()), func(_, offset, size int64) {
		t.impl.FP32Mul(
			t.data[offset:offset+size],
			t2.data[offset:offset+size],
			data[offset:offset+size])
	})
	return NewFloat32(data, append(size1, d1),
		gomath.WithDevice(t.Device()))
}

func (*Float32) mulScalar(t *Float32, scalar float32) gomath.Tensor {
	size, d := splitSize1(t)
	head := count(size)
	data := make([]float32, head*d)
	parallel(head*d, int64(runtime.NumCPU()), func(_, offset, size int64) {
		t.impl.FP32MulScalar(t.data[offset:offset+size], scalar, data[offset:offset+size])
	})
	return NewFloat32(data, append(size, d),
		gomath.WithDevice(t.Device()))
}

func (*Float32) mulVector(t *Float32, vector []float32, counts int64) gomath.Tensor {
	data := make([]float32, len(t.data))
	n := int64(len(vector))
	parallel(counts, int64(runtime.NumCPU()), func(_, offset, size int64) {
		for i := offset; i < offset+size; i++ {
			offset := i * n
			t.impl.FP32Mul(t.data[offset:offset+n], vector, data[offset:offset+n])
		}
	})
	return NewFloat32(data, t.Size(),
		gomath.WithDevice(t.Device()))
}

func (t *Float32) Div(t2 gomath.Tensor) gomath.Tensor {
	panic("implement me")
}
