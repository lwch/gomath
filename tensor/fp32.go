package tensor

import (
	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
	"github.com/lwch/gomath/internal/half"
	"github.com/lwch/gomath/internal/tensor"
	"github.com/lwch/gomath/internal/tensor/ctensor"
	"github.com/lwch/gomath/internal/tensor/gotensor"
)

type Float32 struct {
	*tensor.Tensor
	store *Float32Storage
	impl  tensor.TensorImpl
}

var _ gomath.Tensor = &Float32{}

func NewFloat32(data []float32, shape []int64, opts ...tensor.Option) *Float32 {
	args := tensor.DefaultOptions()
	for _, opt := range opts {
		opt(args)
	}

	var ret Float32
	ret.Tensor = tensor.New(args.Device, shape...)
	ret.store = newFloat32Storage(int(sumShapes(shape)), shape[len(shape)-1])
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

func (t *Float32) Type() consts.Type {
	return consts.Float32
}

func (t *Float32) Storage() gomath.Storage {
	return t.store
}

func (t *Float32) Add(t2 gomath.Tensor) gomath.Tensor {
	panic("implement me")
}

func (t *Float32) Sub(t2 gomath.Tensor) gomath.Tensor {
	panic("implement me")
}

func (t *Float32) Transpose() gomath.Tensor {
	panic("implement me")
}

func (t *Float32) Reshape(shape []int64) gomath.Tensor {
	panic("implement me")
}

func (t *Float32) View(shape []int64) gomath.Tensor {
	panic("implement me")
}

func convertFloat32ToFloat16(t *Float32) gomath.Tensor {
	data := make([]uint16, t.store.Size())
	t.store.Range(func(i int, a any) {
		data[i] = half.Encode(a.(float32))
	})
	return NewFloat16Raw(data, t.Size(),
		gomath.WithDevice(t.Device()))
}
