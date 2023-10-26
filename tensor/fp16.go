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

type Float16 struct {
	*tensor.Tensor
	data []uint16
	impl tensor.TensorImpl
}

var _ gomath.Tensor = &Float16{}

func NewFloat16(data []float32, shape []int64, opts ...tensor.Option) *Float16 {
	args := tensor.DefaultOptions()
	for _, opt := range opts {
		opt(args)
	}

	var ret Float16
	ret.Tensor = tensor.New(args.Device, shape...)
	ret.data = make([]uint16, len(data))
	for i, v := range data {
		ret.data[i] = half.Encode(v)
	}
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

func NewFloat16Raw(data []uint16, shape []int64, opts ...tensor.Option) *Float16 {
	args := tensor.DefaultOptions()
	for _, opt := range opts {
		opt(args)
	}

	var ret Float16
	ret.Tensor = tensor.New(args.Device, shape...)
	ret.data = data
	return &ret
}

func (t *Float16) Type() consts.Type {
	return consts.Float16
}

func (t *Float16) Storage() gomath.Storage {
	return t.data
}

func (t *Float16) Add(t2 gomath.Tensor) gomath.Tensor {
	panic("implement me")
}

func (t *Float16) Sub(t2 gomath.Tensor) gomath.Tensor {
	panic("implement me")
}

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

func (t *Float16) Div(t2 gomath.Tensor) gomath.Tensor {
	panic("implement me")
}

func (t *Float16) MatMul(t2 gomath.Tensor) gomath.Tensor {
	switch t2.Type() {
	case consts.Float16:
		return t.matMul(t2.(*Float16))
	case consts.Float32:
		return convert(t, consts.Float32).MatMul(t2)
	default:
		panic(ErrNotSupported)
	}
}

func (t *Float16) matMul(t2 *Float16) gomath.Tensor {
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
	data := make([]uint16, head*m*n)
	core := runtime.NumCPU()
	parallel(head, int64(core), func(_, offset, size int64) {
		for block := offset; block < offset+size; block++ {
			offset1 := block * m * d1
			offset2 := block * n * d1
			idx := block * m * n
			parallel(m, int64(core), func(_, offset, size int64) {
				for rows := offset; rows < offset+size; rows++ {
					offset1 := offset1 + rows*d1
					idx := idx + rows*n
					for cols := int64(0); cols < n; cols++ {
						offset2 := offset2 + cols*d1
						data[idx+cols] = t.impl.FP16DotVector(
							t.data[offset1:offset1+d1],
							t2.data[offset2:offset2+d1])
					}
				}
			})
		}
	})
	return NewFloat16Raw(data, append(size1, m, n),
		gomath.WithDevice(t.Device()))
}

func (t *Float16) Transpose() gomath.Tensor {
	panic("implement me")
}

func (t *Float16) Reshape(shape []int64) gomath.Tensor {
	panic("implement me")
}

func (t *Float16) View(shape []int64) gomath.Tensor {
	panic("implement me")
}

func convertFloat16ToFloat32(t *Float16) *Float32 {
	data := make([]float32, len(t.data))
	for i, v := range t.data {
		data[i] = half.Decode(v)
	}
	return NewFloat32(data, t.Size(),
		gomath.WithDevice(t.Device()))
}
