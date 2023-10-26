package tensor

import (
	"fmt"
	"runtime"

	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
	"github.com/lwch/gomath/internal/half"
	"github.com/lwch/gomath/internal/tensor"
	"github.com/lwch/gomath/internal/tensor/ctensor"
	"github.com/lwch/gomath/internal/tensor/gotensor"
)

type Float32 struct {
	*tensor.Tensor
	data []float32
	impl tensor.TensorImpl
}

var _ gomath.Tensor = &Float32{}

func NewFloat32(data []float32, shape []int64, opts ...tensor.Option) *Float32 {
	args := tensor.DefaultOptions()
	for _, opt := range opts {
		opt(args)
	}

	var ret Float32
	ret.Tensor = tensor.New(args.Device, shape...)
	ret.data = data
	if debug {
		ret.impl = gotensor.New()
	} else {
		var ok bool
		ret.impl, ok = ctensor.New()
		if !ok {
			ret.impl = gotensor.New()
		}
	}
	return &ret
}

func (t *Float32) Type() consts.Type {
	return consts.Float32
}

func (t *Float32) Storage() gomath.Storage {
	return t.data
}

func (t *Float32) Add(t2 gomath.Tensor) gomath.Tensor {
	panic("implement me")
}

func (t *Float32) Sub(t2 gomath.Tensor) gomath.Tensor {
	panic("implement me")
}

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
	if head == 0 {
		head = 1
	}
	data := make([]float32, head*d)
	parallel(head*d, int64(runtime.NumCPU()), func(_, offset, size int64) {
		t.impl.FP32MulScalar(t.data[offset:offset+size], scalar, data[offset:offset+size])
	})
	return NewFloat32(data, append(size, d),
		gomath.WithDevice(t.Device()))
}

func (t *Float32) Div(t2 gomath.Tensor) gomath.Tensor {
	panic("implement me")
}

// MatMul (..., m, d) @ (..., n, d) -> (..., m, n)
func (t *Float32) MatMul(t2 gomath.Tensor) gomath.Tensor {
	switch t2.Type() {
	case consts.Float16:
		return t.matMul(convert(t2, consts.Float32).(*Float32))
	case consts.Float32:
		return t.matMul(t2.(*Float32))
	default:
		panic("not supported")
	}
}

func (t *Float32) matMul(t2 *Float32) gomath.Tensor {
	size1, m, d1 := splitSize2(t)
	size2, n, d2 := splitSize2(t2)
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
	data := make([]float32, head*m*n)
	core := runtime.NumCPU()
	parallel(head, int64(core), func(_, offset, size int64) {
		for block := offset; block < offset+size; block++ {
			offset1 := block * m * d1
			offset2 := block * n * d2
			idx := block * m * n
			parallel(m, int64(core), func(batch, offset, size int64) {
				for rows := offset; rows < offset+size; rows++ {
					offset1 := offset1 + rows*d1
					idx := idx + rows*n
					for cols := int64(0); cols < n; cols++ {
						offset2 := offset2 + cols*d2
						data[idx+cols] = t.impl.FP32DotVector(
							t.data[offset1:offset1+d1],
							t2.data[offset2:offset2+d2])
					}
				}
			})
		}
	})
	return NewFloat32(data, append(size1, m, n),
		gomath.WithDevice(t.Device()))
}

func (t *Float32) Transpose() gomath.Tensor {
	panic("implement me")
}

func (t *Float32) Reshape(shape []int64) gomath.Tensor {
	panic("implement me")
}

func (t *Float32) View(shape []int64) gomath.Tensor {
	panic("implement me")
}

func convertFloat32ToFloat16(t *Float32) gomath.Tensor {
	data := make([]uint16, len(t.data))
	for i, v := range t.data {
		data[i] = half.Encode(v)
	}
	return NewFloat16Raw(data, t.Size(),
		gomath.WithDevice(t.Device()))
}
