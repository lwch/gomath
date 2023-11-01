package tensor

import (
	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
)

func convert(t gomath.Tensor, dtype consts.Type) gomath.Tensor {
	switch t := t.(type) {
	case *Float16:
		return convertFloat16(t, dtype)
	case *Float32:
		return convertFloat32(t, dtype)
	case *View:
		return convertView(t, dtype)
	default:
		panic(ErrNotSupported)
	}
}

func convertFloat16(t *Float16, dtype consts.Type) gomath.Tensor {
	switch dtype {
	case consts.Float16:
		return t
	case consts.Float32:
		return convertFloat16ToFloat32(t)
	}
	panic("not support")
}

func convertFloat32(t *Float32, dtype consts.Type) gomath.Tensor {
	switch dtype {
	case consts.Float16:
		return convertFloat32ToFloat16(t)
	case consts.Float32:
		return t
	}
	panic("not support")
}

func convertView(t *View, dtype consts.Type) gomath.Tensor {
	if t.Type() == dtype {
		return t
	}
	switch dtype {
	case consts.Float16:
		return convertViewToFloat16(t)
	case consts.Float32:
		return convertViewToFloat32(t)
	}
	panic("not support")
}
