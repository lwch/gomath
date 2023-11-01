package tensor

import (
	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
	"github.com/lwch/gomath/internal/half"
	"github.com/lwch/gomath/internal/tensor"
)

type View struct {
	device consts.Device
	shape  []int64
	stride []int64
	store  gomath.Storage
}

var _ gomath.Tensor = &View{}

func (v *View) Device() consts.Device {
	return v.device
}

func (v *View) ToDevice(device consts.Device) {
	// TODO: implement
	v.device = device
}

func (v *View) ToType(t consts.Type) gomath.Tensor {
	return convert(v, t)
}

func (v *View) Type() consts.Type {
	return v.store.Type()
}

func (v *View) Dim() int64 {
	return int64(len(v.shape))
}

func (v *View) Size() []int64 {
	return v.shape
}

func (v *View) Stride() []int64 {
	return v.stride
}

func (v *View) ElemSize() int64 {
	return sumShapes(v.shape)
}

func (v *View) Add(t gomath.Tensor) gomath.Tensor {
	return v.do2(t, addFP16, addFP32)
}

func (v *View) AddScalar(s float32) gomath.Tensor {
	return v.doScalar(s, scalarAddFP16, scalarAddFP32)
}

func (v *View) Sub(t gomath.Tensor) gomath.Tensor {
	return v.do2(t, subFP16, subFP32)
}

func (v *View) SubScalar(s float32) gomath.Tensor {
	return v.doScalar(-s, scalarAddFP16, scalarAddFP16)
}

func (v *View) Mul(t gomath.Tensor) gomath.Tensor {
	return v.do2(t, mulFP16, mulFP32)
}

func (v *View) MulScalar(s float32) gomath.Tensor {
	return v.doScalar(s, scalarMulFP16, scalarMulFP32)
}

func (v *View) Div(t gomath.Tensor) gomath.Tensor {
	return v.do2(t, divFP16, divFP32)
}

func (v *View) DivScalar(s float32) gomath.Tensor {
	return v.doScalar(1/s, scalarMulFP16, scalarMulFP32)
}

func (v *View) MatMul(t gomath.Tensor) gomath.Tensor {
	return v.do2(t, matMulFP16, matMulFP32)
}

func (v *View) Transpose(dim0, dim1 int64) gomath.Tensor {
	return transpose(v, dim0, dim1)
}

func (v *View) Reshape(shape ...int64) gomath.Tensor {
	return reshape(v, shape)
}

func (v *View) View(shape ...int64) gomath.Tensor {
	return view(v, shape)
}

func (v *View) Storage() gomath.Storage {
	return v.store
}

func (t *View) Get(idx ...int64) any {
	return get(idx, t.Stride(), t.store)
}

func (t *View) IsContiguous() bool {
	return isContiguous(t.Size(), t.Stride())
}

func (t *View) Contiguous() gomath.Tensor {
	if t.IsContiguous() {
		return t
	}
	return contiguous(t)
}

type method2 func(gomath.Tensor, gomath.Tensor) gomath.Tensor

func (v *View) do2(t gomath.Tensor, fnFP16, fnFP32 method2) gomath.Tensor {
	switch v.Type() {
	case consts.Float16:
		if t.Type() == consts.Float16 {
			return fnFP16(v, t)
		} else {
			return convert(v, consts.Float32).MatMul(t)
		}
	case consts.Float32:
		if t.Type() == consts.Float32 {
			return fnFP32(v, t)
		} else {
			return convert(v, consts.Float32).MatMul(t)
		}
	default:
		panic(ErrNotSupported)
	}
}

type methodScalar func(any, gomath.Tensor) gomath.Tensor

func (v *View) doScalar(n float32, fnFP16, fnFP32 methodScalar) gomath.Tensor {
	switch v.Type() {
	case consts.Float16:
		return fnFP16(half.Encode(n), v)
	case consts.Float32:
		return fnFP32(n, v)
	default:
		panic(ErrNotSupported)
	}
}

func convertViewToFloat16(t *View) gomath.Tensor {
	if t.Type() != consts.Float32 {
		panic(ErrNotSupported)
	}
	data := make([]uint16, t.ElemSize())
	for i := int64(0); i < t.ElemSize(); i++ {
		data[i] = half.Encode(t.store.Get(i).(float32))
	}
	var ret Float16
	ret.Tensor = tensor.Raw(t.device, t.shape, t.stride)
	ret.store = NewFloat16Storage(data)
	return &ret
}

func convertViewToFloat32(t *View) *Float32 {
	if t.Type() != consts.Float16 {
		panic(ErrNotSupported)
	}
	data := make([]float32, t.ElemSize())
	for i := int64(0); i < t.ElemSize(); i++ {
		data[i] = half.Decode(t.store.Get(i).(uint16))
	}
	var ret Float32
	ret.Tensor = tensor.Raw(t.device, t.shape, t.stride)
	ret.store = NewFloat32Storage(data)
	return &ret
}
