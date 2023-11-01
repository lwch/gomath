package tensor

import (
	"github.com/lwch/gomath/consts"
	"github.com/lwch/gomath/internal/utils"
)

type Tensor struct {
	device consts.Device
	shape  []int64
	stride []int64
}

func New(device consts.Device, shape ...int64) *Tensor {
	return &Tensor{
		device: device,
		shape:  shape,
		stride: utils.ComputeStride(shape),
	}
}

func Raw(device consts.Device, shape, stride []int64) *Tensor {
	return &Tensor{
		device: device,
		shape:  shape,
		stride: stride,
	}
}

func (t *Tensor) Device() consts.Device {
	return t.device
}

func (t *Tensor) Dim() int64 {
	return int64(len(t.shape))
}

func (t *Tensor) Size() []int64 {
	return t.shape
}

func (t *Tensor) Stride() []int64 {
	return t.stride
}

func (t *Tensor) ElemSize() int64 {
	size := t.shape[0]
	for i := 1; i < len(t.shape); i++ {
		size *= t.shape[i]
	}
	return size
}

func (t *Tensor) ToDevice(device consts.Device) {
	// TODO: implement
	t.device = device
}
