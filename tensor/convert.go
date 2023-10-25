package tensor

import (
	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
)

func convert(t gomath.Tensor, dtype consts.Type) gomath.Tensor {
	switch t.Type() {
	case consts.Float16:
		return convertFloat16(t.(*Float16), dtype)
	case consts.Float32:
		return convertFloat32(t.(*Float32), dtype)
	default:
		panic("not supported")
	}
}

func convertFloat16(t *Float16, dtype consts.Type) gomath.Tensor {
	switch dtype {
	case consts.Float16:
		panic("no need")
	case consts.Float32:
		return convertFloat16ToFloat32(t)
	case consts.Float64:
		return convertFloat16ToFloat64(t)
	}
	panic("not support")
}

func convertFloat32(t *Float32, dtype consts.Type) gomath.Tensor {
	switch dtype {
	case consts.Float16:
		return convertFloat32ToFloat16(t)
	case consts.Float32:
		panic("no need")
	case consts.Float64:
		return convertFloat32ToFloat64(t)
	}
	panic("not support")
}
