package tensor

import (
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

func (t *Float32) MatMul(t2 gomath.Tensor) gomath.Tensor {
	panic("implement me")
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
