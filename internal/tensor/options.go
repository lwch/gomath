package tensor

import "github.com/lwch/gomath/consts"

type Options struct {
	Device consts.Device
}

func DefaultOptions() *Options {
	return &Options{
		Device: consts.CPU,
	}
}

type Option func(*Options)
