package f32

import (
	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
	"github.com/lwch/gomath/internal/tensor"
)

type Float32 struct {
	*tensor.Tensor
	data []Float32
}

var _ gomath.Tensor = &Float32{}

func New(data []Float32, shape []int64, opts ...tensor.Option) *Float32 {
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
