package gomath

import (
	"github.com/lwch/gomath/consts"
	"github.com/lwch/gomath/internal/tensor"
)

func WithDevice(device consts.Device) tensor.Option {
	return func(o *tensor.Options) {
		o.Device = device
	}
}
