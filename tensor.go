package gomath

import (
	"github.com/lwch/gomath/consts"
	"github.com/lwch/gomath/internal/tensor"
)

type Tensor interface {
	Device() consts.Device
	Type() consts.Type
	Size() []int64
	Stride() []int64
	Add(Tensor) Tensor
	Sub(Tensor) Tensor
	Mul(Tensor) Tensor
	Div(Tensor) Tensor
	MatMul(Tensor) Tensor
	Transpose() Tensor
	Reshape([]int64) Tensor
	View([]int64) Tensor
	Storage() Storage
}

type Storage interface {
}

func WithDevice(device consts.Device) tensor.Option {
	return func(o *tensor.Options) {
		o.Device = device
	}
}
