package gotensor

import (
	"github.com/chewxy/math32"
	"github.com/lwch/gomath/internal/half"
)

func (*GoTensor) FP16Dot(x, w []uint16) uint16 {
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

func (*GoTensor) FP16ScalarMul(x uint16, w, y []uint16) {
	dx := half.Decode(x)
	for i := range w {
		y[i] = half.Encode(dx * half.Decode(w[i]))
	}
}

func (*GoTensor) FP16Div(x, w, y []uint16) {
	for i := range x {
		y[i] = half.Encode(half.Decode(x[i]) / half.Decode(w[i]))
	}
}

func (*GoTensor) FP16ScalarDiv(x uint16, w, y []uint16) {
	dx := half.Decode(x)
	for i := range w {
		y[i] = half.Encode(dx / half.Decode(w[i]))
	}
}

func (*GoTensor) FP16Add(x, w, y []uint16) {
	for i := range x {
		y[i] = half.Encode(half.Decode(x[i]) + half.Decode(w[i]))
	}
}

func (*GoTensor) FP16ScalarAdd(x uint16, w, y []uint16) {
	dx := half.Decode(x)
	for i := range w {
		y[i] = half.Encode(dx + half.Decode(w[i]))
	}
}

func (*GoTensor) FP16Sub(x, w, y []uint16) {
	for i := range x {
		y[i] = half.Encode(half.Decode(x[i]) - half.Decode(w[i]))
	}
}

func (*GoTensor) FP16ScalarSub(x uint16, w, y []uint16) {
	dx := half.Decode(x)
	for i := range w {
		y[i] = half.Encode(dx - half.Decode(w[i]))
	}
}

func (*GoTensor) FP16Pow(x []uint16, n uint16, y []uint16) {
	dn := half.Decode(n)
	for i := range x {
		y[i] = half.Encode(math32.Pow(half.Decode(x[i]), dn))
	}
}
