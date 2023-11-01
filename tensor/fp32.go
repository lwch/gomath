package tensor

import (
	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
	"github.com/lwch/gomath/internal/half"
	"github.com/lwch/gomath/internal/tensor"
)

type Float32 struct {
	*tensor.Tensor
	store Float32Storage
}

var _ gomath.Tensor = &Float32{}

func NewFloat32WithStorage(s Float32Storage, shape []int64, opts ...tensor.Option) *Float32 {
	args := tensor.DefaultOptions()
	for _, opt := range opts {
		opt(args)
	}

	var ret Float32
	ret.Tensor = tensor.New(args.Device, shape...)
	ret.store = s
	return &ret
}

func NewFloat32(data []float32, shape []int64, opts ...tensor.Option) *Float32 {
	cvt := make([]float32, sumShapes(shape))
	copy(cvt, data)
	return NewFloat32WithStorage(NewFloat32Storage(cvt), shape, opts...)
}

func (t *Float32) Type() consts.Type {
	return consts.Float32
}

func (t *Float32) Storage() gomath.Storage {
	return t.store
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

func (t *Float32) ToType(tp consts.Type) gomath.Tensor {
	return convert(t, tp)
}

func (t *Float32) row(i int64) any {
	d := t.Size()[t.Dim()-1]
	return []float32(t.store[i*d : (i+1)*d])
}

func convertFloat32ToFloat16(t *Float32) gomath.Tensor {
	data := make([]uint16, t.store.Size())
	t.store.Range(func(i int, a any) {
		data[i] = half.Encode(a.(float32))
	})
	return NewFloat16Raw(data, t.Size(),
		gomath.WithDevice(t.Device()))
}
