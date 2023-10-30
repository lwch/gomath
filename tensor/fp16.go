package tensor

import (
	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
	"github.com/lwch/gomath/internal/half"
	"github.com/lwch/gomath/internal/tensor"
	"github.com/lwch/gomath/internal/tensor/ctensor"
	"github.com/lwch/gomath/internal/tensor/gotensor"
)

type Float16 struct {
	*tensor.Tensor
	store *Float16Storage
	impl  tensor.TensorImpl
}

var _ gomath.Tensor = &Float16{}

func NewFloat16(data []float32, shape []int64, opts ...tensor.Option) *Float16 {
	args := tensor.DefaultOptions()
	for _, opt := range opts {
		opt(args)
	}

	var ret Float16
	ret.Tensor = tensor.New(args.Device, shape...)
	ret.store = newFloat16Storage(int(sumShapes(shape)), shape[len(shape)-1])
	ret.store.Copy(data)
	if debug {
		ret.impl = gotensor.New()
	} else {
		var ok bool
		ret.impl, ok = ctensor.New()
		if !ok {
			ret.impl = gotensor.New()
		}
	}
	return &ret
}

func NewFloat16Raw(data []uint16, shape []int64, opts ...tensor.Option) *Float16 {
	args := tensor.DefaultOptions()
	for _, opt := range opts {
		opt(args)
	}

	var ret Float16
	ret.Tensor = tensor.New(args.Device, shape...)
	ret.store = newFloat16Storage(int(sumShapes(shape)), shape[len(shape)-1])
	ret.store.Copy(data)
	if debug {
		ret.impl = gotensor.New()
	} else {
		var ok bool
		ret.impl, ok = ctensor.New()
		if !ok {
			ret.impl = gotensor.New()
		}
	}
	return &ret
}

func (t *Float16) Type() consts.Type {
	return consts.Float16
}

func (t *Float16) Storage() gomath.Storage {
	return t.store
}

func (t *Float16) Add(t2 gomath.Tensor) gomath.Tensor {
	panic("implement me")
}

func (t *Float16) Sub(t2 gomath.Tensor) gomath.Tensor {
	panic("implement me")
}

func (t *Float16) Transpose() gomath.Tensor {
	panic("implement me")
}

func (t *Float16) Reshape(shape []int64) gomath.Tensor {
	panic("implement me")
}

func (t *Float16) View(shape []int64) gomath.Tensor {
	panic("implement me")
}

func convertFloat16ToFloat32(t *Float16) *Float32 {
	data := make([]float32, t.store.Size())
	t.store.Range(func(i int, a any) {
		data[i] = half.Decode(a.(uint16))
	})
	return NewFloat32(data, t.Size(),
		gomath.WithDevice(t.Device()))
}
