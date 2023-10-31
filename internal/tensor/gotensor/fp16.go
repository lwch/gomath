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

func (*GoTensor) FP16MulScalar(x uint16, w, y []uint16) {
	dx := half.Decode(x)
	for i := range w {
		y[i] = half.Encode(dx * half.Decode(w[i]))
	}
}

func (*GoTensor) FP16MulVector(x, w, y []uint16) {
	for i := range x {
		y[i] = half.Encode(half.Decode(x[i]) * half.Decode(w[i]))
	}
}

func (*GoTensor) FP16DivScalar(x uint16, w, y []uint16) {
	dx := half.Decode(x)
	for i := range w {
		y[i] = half.Encode(dx / half.Decode(w[i]))
	}
}

func (*GoTensor) FP16DivVector(x, w, y []uint16) {
	for i := range x {
		y[i] = half.Encode(half.Decode(x[i]) / half.Decode(w[i]))
	}
}

func (*GoTensor) FP16AddScalar(x []uint16, w uint16, y []uint16) {
	dw := half.Decode(w)
	for i := range x {
		y[i] = half.Encode(half.Decode(x[i]) + dw)
	}
}

func (*GoTensor) FP16AddVector(x, w, y []uint16) {
	for i := range x {
		y[i] = half.Encode(half.Decode(x[i]) + half.Decode(w[i]))
	}
}
