package tensor

// #include "fp16.h"
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
	"github.com/lwch/gomath/internal/half"
	"github.com/lwch/gomath/internal/tensor"
)

type Float16 struct {
	*tensor.Tensor
	data []uint16
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
	panic("implement me")
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
		panic("not supported")
	}
}

func (t *Float16) matMul(t2 *Float16) gomath.Tensor {
	size1, m, d1 := splitSize(t)
	size2, n, d2 := splitSize(t2)
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
	if computeInC(consts.Float32) {
		return t.cMatMul(t2, size1, head, m, n, d1)
	}
	return t.goMatMul(t2, size1, head, m, n, d1)
}

func (t *Float16) cMatMul(t2 *Float16, size []int64, head, m, n, d int64) gomath.Tensor {
	data := make([]uint16, head*m*n)
	core := int64(runtime.NumCPU())
	p1 := unsafe.Pointer(unsafe.SliceData(t.data))
	p2 := unsafe.Pointer(unsafe.SliceData(t2.data))
	parallel(head, core, func(_, offset, size int64) {
		for block := offset; block < offset+size; block++ {
			offset1 := block * m * d
			offset2 := block * n * d
			idx := block * m * n
			parallel(m, core, func(_, offset, size int64) {
				for rows := offset; rows < offset+size; rows++ {
					offset1 := offset1 + rows*d
					for cols := int64(0); cols < n; cols++ {
						offset2 := offset2 + cols*d
						data[idx] = uint16(C.fp16_dot_vector(
							(*C.uint16_t)(unsafe.Add(p1, uintptr(offset1*2))),
							(*C.uint16_t)(unsafe.Add(p2, uintptr(offset2*2))),
							C.int64_t(d)))
						idx++
					}
				}
			})
		}
	})
	return NewFloat16Raw(data, append(size, m, n),
		gomath.WithDevice(t.Device()))
}

func (t *Float16) goMatMul(t2 *Float16, size []int64, head, m, n, d int64) gomath.Tensor {
	data := make([]uint16, head*m*n)
	core := int64(runtime.NumCPU())
	parallel(head, core, func(_, offset, size int64) {
		for block := offset; block < offset+size; block++ {
			offset1 := block * m * d
			offset2 := block * n * d
			idx := block * m * n
			parallel(m, core, func(_, offset, size int64) {
				data1 := make([]float32, d)
				data2 := make([]float32, d)
				for rows := offset; rows < offset+size; rows++ {
					offset1 := offset1 + rows*d
					idx := idx + rows*n
					for cols := int64(0); cols < n; cols++ {
						offset2 := offset2 + cols*d
						half.DecodeArray(t.data[offset1:offset1+d], data1)
						half.DecodeArray(t2.data[offset2:offset2+d], data2)
						data[idx+cols] = half.Encode(dotVector(data1, data2, d))
					}
				}
			})
		}
	})
	return NewFloat16Raw(data, append(size, m, n),
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
