package tensor

//#include "bf16.h"
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
	"github.com/lwch/gomath/internal/tensor"
)

type BFloat16 struct {
	*tensor.Tensor
	data []uint16
}

var _ gomath.Tensor = &BFloat16{}

func NewBFloat16(data []uint16, shape []int64, opts ...tensor.Option) *BFloat16 {
	args := tensor.DefaultOptions()
	for _, opt := range opts {
		opt(args)
	}

	var ret BFloat16
	ret.Tensor = tensor.New(args.Device, shape...)
	ret.data = data
	return &ret
}

func (t *BFloat16) Type() consts.Type {
	return consts.BFloat16
}

func (t *BFloat16) Storage() gomath.Storage {
	return t.data
}

func (t *BFloat16) Add(t2 gomath.Tensor) gomath.Tensor {
	panic("implement me")
}

func (t *BFloat16) Sub(t2 gomath.Tensor) gomath.Tensor {
	panic("implement me")
}

func (t *BFloat16) Mul(t2 gomath.Tensor) gomath.Tensor {
	panic("implement me")
}

func (t *BFloat16) Div(t2 gomath.Tensor) gomath.Tensor {
	panic("implement me")
}

// MatMul (..., m, d) @ (..., n, d) -> (..., m, n)
func (t *BFloat16) MatMul(t2 gomath.Tensor) gomath.Tensor {
	switch t2.Type() {
	case consts.BFloat16:
		return t.matMulBFloat16(t2.(*BFloat16))
	case consts.Float32:
		return convert(t, consts.Float32).MatMul(t2)
	default:
		panic("not supported")
	}
}

func (t *BFloat16) matMulBFloat16(t2 *BFloat16) gomath.Tensor {
	size1, m, d1 := splitSize(t)
	size2, n, d2 := splitSize(t2)
	if d1 != d2 {
		panic(fmt.Errorf("dimension mismatch: %v and %v", t.Size(), t2.Size()))
	}
	if !sizeMatch(size1, size2) {
		// TODO: broadcast
		panic(fmt.Errorf("size mismatch: %v and %v", t.Size(), t2.Size()))
	}
	head := count(size1)
	if head == 0 {
		head = 1
	}
	targetSize := head * m * n
	data := make([]uint16, targetSize)
	if computeInC() {
		data1 := make([]float32, d1)
		data2 := make([]float32, d2)
		for block := int64(0); block < head; block++ {
			offset1 := block * m * d1
			offset2 := block * n * d2
			for rows := int64(0); rows < m; rows++ {
				offset1 = offset1 + rows*d1
				offset2 = offset2 + rows*d2
				for cols := int64(0); cols < n; cols++ {
					idx := block*m*n + rows*n + cols
					dx := C.bf16_dot_vector(
						(*C.float)(unsafe.SliceData(data1)),
						(*C.float)(unsafe.SliceData(data2)),
						C.int64_t(d1))
					data[idx] = float32ToBF16(float32(dx))
				}
			}
		}
	}
	return NewBFloat16(data, append(size1, m, n),
		gomath.WithDevice(t.Device()))
}

func (t *BFloat16) Transpose() gomath.Tensor {
	panic("implement me")
}

func (t *BFloat16) Reshape(shape []int64) gomath.Tensor {
	panic("implement me")
}

func (t *BFloat16) View(shape []int64) gomath.Tensor {
	panic("implement me")
}

func bf16ToFloat32(u16 uint16) float32 {
	u32 := uint32(u16) << 16
	return *(*float32)(unsafe.Pointer(&u32))
}

func convertBFloat16ToFloat32(t *BFloat16) *Float32 {
	data := make([]float32, len(t.data))
	for i, v := range t.data {
		data[i] = bf16ToFloat32(v)
	}
	return NewFloat32(data, t.Size(),
		gomath.WithDevice(t.Device()))
}
