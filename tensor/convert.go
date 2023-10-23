package tensor

import (
	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
)

func convert(t gomath.Tensor, dtype consts.Type) gomath.Tensor {
	switch t.Type() {
	case consts.BFloat16:
		return convertBFloat16(t.(*BFloat16), dtype)
	case consts.Float32:
		return convertFloat32(t.(*Float32), dtype)
	default:
		panic("not supported")
	}
}

func convertBFloat16(t *BFloat16, dtype consts.Type) gomath.Tensor {
	switch dtype {
	case consts.Float32:
		return convertBFloat16ToFloat32(t)
	}
	panic("not support")
}

func convertFloat32(t *Float32, dtype consts.Type) gomath.Tensor {
	switch dtype {
	case consts.BFloat16:
		return convertFloat32ToBFloat16(t)
	}
	panic("not support")
}
