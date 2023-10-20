package bf16

import (
	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
	"github.com/lwch/gomath/internal/tensor"
)

type BFloat16 struct {
	*tensor.Tensor
	data []uint16
}

var _ gomath.Tensor = &BFloat16{}

func New(data []uint16, shape []int64, opts ...tensor.Option) *BFloat16 {
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

func (t *BFloat16) MatMul(t2 gomath.Tensor) gomath.Tensor {
	panic("implement me")
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