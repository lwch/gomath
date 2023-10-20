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

func (t *Tensor) Device() consts.Device {
	return t.device
}

func (t *Tensor) Size() []int64 {
	return t.shape
}

func (t *Tensor) Stride() []int64 {
	return t.stride
}
