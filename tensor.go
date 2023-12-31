package gomath

import (
	"github.com/lwch/gomath/consts"
)

type Tensor interface {
	Device() consts.Device
	ToDevice(consts.Device)
	ToType(consts.Type) Tensor
	Type() consts.Type
	Dim() int64
	Size() []int64
	Stride() []int64
	ElemSize() int64
	Add(Tensor) Tensor
	AddScalar(float32) Tensor
	Sub(Tensor) Tensor
	SubScalar(float32) Tensor
	Mul(Tensor) Tensor
	MulScalar(float32) Tensor
	Div(Tensor) Tensor
	DivScalar(float32) Tensor
	MatMul(Tensor) Tensor
	Pow(float32) Tensor
	Transpose(int64, int64) Tensor
	Reshape(...int64) Tensor
	View(...int64) Tensor
	Get(...int64) any
	Storage() Storage
	IsContiguous() bool
	Contiguous() Tensor
}

type Storage interface {
	Type() consts.Type
	Data() any
	Size() int
	Get(int64) any
	Set(int64, any)
	Range(int64, int64) any
	Loop(func(int, any))
}
