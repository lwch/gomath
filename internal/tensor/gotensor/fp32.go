package gotensor

import "github.com/chewxy/math32"

func (*GoTensor) FP32Dot(x, w []float32) float32 {
	return dotVector(x, w, int64(len(x)))
}

func (*GoTensor) FP32Mul(x, w, y []float32) {
	for i := range x {
		y[i] = x[i] * w[i]
	}
}

func (*GoTensor) FP32ScalarMul(x float32, w, y []float32) {
	for i := range w {
		y[i] = x * w[i]
	}
}

func (*GoTensor) FP32Div(x, w, y []float32) {
	for i := range x {
		y[i] = x[i] / w[i]
	}
}

func (*GoTensor) FP32ScalarDiv(x float32, w, y []float32) {
	for i := range w {
		y[i] = x / w[i]
	}
}

func (*GoTensor) FP32Add(x, w, y []float32) {
	for i := range x {
		y[i] = x[i] + w[i]
	}
}

func (*GoTensor) FP32ScalarAdd(x float32, w, y []float32) {
	for i := range w {
		y[i] = x + w[i]
	}
}

func (*GoTensor) FP32Sub(x, w, y []float32) {
	for i := range x {
		y[i] = x[i] - w[i]
	}
}

func (*GoTensor) FP32ScalarSub(x float32, w, y []float32) {
	for i := range w {
		y[i] = x - w[i]
	}
}

func (*GoTensor) FP32Pow(x []float32, n float32, y []float32) {
	for i := range x {
		y[i] = math32.Pow(x[i], n)
	}
}
