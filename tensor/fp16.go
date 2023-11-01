package tensor

import (
	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
	"github.com/lwch/gomath/internal/half"
	"github.com/lwch/gomath/internal/tensor"
)

type Float16 struct {
	*tensor.Tensor
	store *Float16Storage
}

var _ gomath.Tensor = &Float16{}

func NewFloat16WithStorage(s *Float16Storage, shape []int64, opts ...tensor.Option) *Float16 {
	args := tensor.DefaultOptions()
	for _, opt := range opts {
		opt(args)
	}

	var ret Float16
	ret.Tensor = tensor.New(args.Device, shape...)
	ret.store = s
	return &ret
}

func NewFloat16(data []float32, shape []int64, opts ...tensor.Option) *Float16 {
	cvt := make([]uint16, sumShapes(shape))
	for i, v := range data {
		cvt[i] = half.Encode(v)
	}
	return NewFloat16WithStorage(NewFloat16Storage(cvt, shape[len(shape)-1]), shape, opts...)
}

func NewFloat16Raw(data []uint16, shape []int64, opts ...tensor.Option) *Float16 {
	cvt := make([]uint16, sumShapes(shape))
	copy(cvt, data)
	return NewFloat16WithStorage(NewFloat16Storage(cvt, shape[len(shape)-1]), shape, opts...)
}

func (t *Float16) Type() consts.Type {
	return consts.Float16
}

func (t *Float16) Storage() gomath.Storage {
	return t.store
}

func (t *Float16) Sub(t2 gomath.Tensor) gomath.Tensor {
	panic("implement me")
}

func (t *Float16) SubScalar(n float32) gomath.Tensor {
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

func (t *Float16) ToType(tp consts.Type) gomath.Tensor {
	return convert(t, tp)
}

func convertFloat16ToFloat32(t *Float16) *Float32 {
	data := make([]float32, t.store.Size())
	t.store.Range(func(i int, a any) {
		data[i] = half.Decode(a.(uint16))
	})
	return NewFloat32(data, t.Size(),
		gomath.WithDevice(t.Device()))
}
