package tensor

import (
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
	panic("implement me")
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

func convertFloat16ToFloat64(t *Float16) *Float64 {
	data := make([]float64, len(t.data))
	for i, v := range t.data {
		data[i] = float64(half.Decode(v))
	}
	return NewFloat64(data, t.Size(),
		gomath.WithDevice(t.Device()))
}
