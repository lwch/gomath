package tensor

import (
	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
	"github.com/lwch/gomath/internal/tensor"
)

type Float16 struct {
	*tensor.Tensor
	data []uint16
}

var _ gomath.Tensor = &Float16{}

func NewFloat16(data []uint16, shape []int64, opts ...tensor.Option) *Float16 {
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
