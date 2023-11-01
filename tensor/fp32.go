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

func (t *Float32) Transpose(dim0, dim1 int64) gomath.Tensor {
	return transpose(t, dim0, dim1)
}

func (t *Float32) Reshape(shape ...int64) gomath.Tensor {
	return reshape(t, shape)
}

func (t *Float32) View(shape ...int64) gomath.Tensor {
	return view(t, shape)
}

func (t *Float32) ToType(tp consts.Type) gomath.Tensor {
	return convert(t, tp)
}

func (t *Float32) Get(idx ...int64) any {
	return get(idx, t.Stride(), t.store)
}

func (t *Float32) IsContiguous() bool {
	return isContiguous(t.Size(), t.Stride())
}

func (t *Float32) Contiguous() gomath.Tensor {
	if t.IsContiguous() {
		return t
	}
	return contiguous(t)
}

func convertFloat32ToFloat16(t *Float32) gomath.Tensor {
	data := make([]uint16, t.store.Size())
	t.store.Range(func(i int, a any) {
		data[i] = half.Encode(a.(float32))
	})
	return NewFloat16Raw(data, t.Size(),
		gomath.WithDevice(t.Device()))
}
