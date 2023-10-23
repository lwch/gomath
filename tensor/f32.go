package tensor

// #include "f32.h"
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
	"github.com/lwch/gomath/internal/tensor"
)

type Float32 struct {
	*tensor.Tensor
	data []float32
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
	panic("implement me")
}

func (t *Float32) Div(t2 gomath.Tensor) gomath.Tensor {
	panic("implement me")
}

// MatMul (..., m, d) @ (..., n, d) -> (..., m, n)
func (t *Float32) MatMul(t2 gomath.Tensor) gomath.Tensor {
	switch t2.Type() {
	case consts.BFloat16:
		return t.MatMul(convert(t2, consts.Float32))
	case consts.Float32:
		return t.matMulFloat32(t2.(*Float32))
	default:
		panic("not supported")
	}
}

func (t *Float32) matMulFloat32(t2 *Float32) gomath.Tensor {
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
	data := make([]float32, targetSize)
	if computeInC() {
		p1 := unsafe.Pointer(unsafe.SliceData(t.data))
		p2 := unsafe.Pointer(unsafe.SliceData(t2.data))
		for block := int64(0); block < head; block++ {
			offset1 := block * m * d1
			offset2 := block * n * d2
			idx := block * m * n
			for rows := int64(0); rows < m; rows++ {
				offset1 := offset1 + rows*d1
				for cols := int64(0); cols < n; cols++ {
					offset2 := offset2 + cols*d2
					data[idx] = float32(C.f32_dot_vector(
						(*C.float)(unsafe.Add(p1, uintptr(offset1*4))),
						(*C.float)(unsafe.Add(p2, uintptr(offset2*4))),
						C.int64_t(d1)))
					idx++
				}
			}
		}
		return NewFloat32(data, append(size1, m, n),
			gomath.WithDevice(t.Device()))
	}
	parallel(head, int64(runtime.NumCPU()), func(_, block, size int64) {
		offset1 := block * m * d1
		offset2 := block * n * d2
		idx := block * m * n
		for rows := int64(0); rows < m; rows++ {
			offset1 := offset1 + rows*d1
			idx := idx + rows*n
			for cols := int64(0); cols < n; cols++ {
				offset2 := offset2 + cols*d2
				data[idx+cols] = dotVector(t.data[offset1:], t2.data[offset2:], d1)
			}
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

func float32ToBF16(f32 float32) uint16 {
	u32 := *(*uint32)(unsafe.Pointer(&f32))
	return uint16(u32 >> 16)
}

func convertFloat32ToBFloat16(t *Float32) *BFloat16 {
	data := make([]uint16, len(t.data))
	for i, v := range t.data {
		data[i] = float32ToBF16(v)
	}
	return NewBFloat16(data, t.Size(),
		gomath.WithDevice(t.Device()))
}