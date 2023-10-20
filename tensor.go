package gomath

import (
	"github.com/lwch/gomath/consts"
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
