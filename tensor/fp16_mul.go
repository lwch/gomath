package tensor

import (
	"fmt"
	"runtime"

	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
)

func (t *Float16) Mul(t2 gomath.Tensor) gomath.Tensor {
	switch t2.Type() {
	case consts.Float16:
		return t.mul(t2.(*Float16))
	case consts.Float32:
		return convert(t, consts.Float32).Mul(t2)
	default:
		panic(ErrNotSupported)
	}
}

func (t *Float16) mul(t2 *Float16) gomath.Tensor {
	size1, d1 := splitSize1(t)
	size2, d2 := splitSize1(t2)
	if len(size1) == 0 && d1 == 1 {
		// scalar * t2
		return t.mulScalar(t2, t.data[0])
	}
	if len(size2) == 0 && d2 == 1 {
		// t * scalar
		return t.mulScalar(t, t2.data[0])
	}
	if d1 != d2 {
		panic(fmt.Errorf("dimension mismatch: %v and %v", t.Size(), t2.Size()))
	}
	if !sizeMatch(size1, size2) {
		panic(ErrBroadcast)
	}
	head := count(size1)
	if head == 0 {
		head = 1
	}
	data := make([]uint16, head*d1)
	parallel(head*d1, int64(runtime.NumCPU()), func(_, offset, size int64) {
		t.impl.FP16Mul(
			t.data[offset:offset+size],
			t2.data[offset:offset+size],
			data[offset:offset+size])
	})
	return NewFloat16Raw(data, append(size1, d1),
		gomath.WithDevice(t.Device()))
}

func (*Float16) mulScalar(t *Float16, scalar uint16) gomath.Tensor {
	data := make([]uint16, len(t.data))
	parallel(int64(len(t.data)), int64(runtime.NumCPU()), func(_, offset, size int64) {
		t.impl.FP16MulScalar(t.data[offset:offset+size], scalar, data[offset:offset+size])
	})
	return NewFloat16Raw(data, t.Size(),
		gomath.WithDevice(t.Device()))
}

func (t *Float16) Div(t2 gomath.Tensor) gomath.Tensor {
	panic("implement me")
}
