package tensor

import (
	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
	"github.com/lwch/gomath/internal/tensor"
)

type Float64 struct {
	*tensor.Tensor
	data []float64
}

var _ gomath.Tensor = &Float64{}

func NewFloat64(data []float64, shape []int64, opts ...tensor.Option) *Float64 {
	args := tensor.DefaultOptions()
	for _, opt := range opts {
		opt(args)
	}

	var ret Float64
	ret.Tensor = tensor.New(args.Device, shape...)
	ret.data = data
	return &ret
}

func (t *Float64) Type() consts.Type {
	return consts.Float64
}

func (t *Float64) Storage() gomath.Storage {
	return t.data
}

func (t *Float64) Add(t2 gomath.Tensor) gomath.Tensor {
	panic("implement me")
}

func (t *Float64) Sub(t2 gomath.Tensor) gomath.Tensor {
	panic("implement me")
}

func (t *Float64) Mul(t2 gomath.Tensor) gomath.Tensor {
	panic("implement me")
}

func (t *Float64) Div(t2 gomath.Tensor) gomath.Tensor {
	panic("implement me")
}

func (t *Float64) MatMul(t2 gomath.Tensor) gomath.Tensor {
	panic("implement me")
}

func (t *Float64) Transpose() gomath.Tensor {
	panic("implement me")
}

func (t *Float64) Reshape(shape []int64) gomath.Tensor {
	panic("implement me")
}

func (t *Float64) View(shape []int64) gomath.Tensor {
	panic("implement me")
}
