package gotensor

import "github.com/lwch/gomath/internal/half"

func (*GoTensor) FP16DotVector(x, w []uint16) uint16 {
	dx := make([]float32, len(x))
	dw := make([]float32, len(w))
	for i := range x {
		dx[i] = half.Decode(x[i])
		dw[i] = half.Decode(w[i])
	}
	return half.Encode(dotVector(dx, dw, int64(len(x))))
}

func (*GoTensor) FP16Mul(x, w, y []uint16) {
	for i := range x {
		y[i] = half.Encode(half.Decode(x[i]) * half.Decode(w[i]))
	}
}
