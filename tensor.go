package gomath

import (
	"github.com/lwch/gomath/consts"
)

type Tensor interface {
	Device() consts.Device
	ToDevice(consts.Device)
	ToType(consts.Type) Tensor
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
	Data() any
	Size() int
	Get(int64) any
	Set(int64, any)
	Range(func(int, any))
	Row(int64) any
	Copy(any)
}
