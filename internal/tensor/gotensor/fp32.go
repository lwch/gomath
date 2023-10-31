package gotensor

func (*GoTensor) FP32DotVector(x, w []float32) float32 {
	return dotVector(x, w, int64(len(x)))
}

func (*GoTensor) FP32MulScalar(x float32, w, y []float32) {
	for i := range w {
		y[i] = x * w[i]
	}
}

func (*GoTensor) FP32MulVector(x, w, y []float32) {
	for i := range x {
		y[i] = x[i] * w[i]
	}
}

func (*GoTensor) FP32DivScalar(x float32, w, y []float32) {
	for i := range w {
		y[i] = x / w[i]
	}
}

func (*GoTensor) FP32DivVector(x, w, y []float32) {
	for i := range x {
		y[i] = x[i] / w[i]
	}
}

func (*GoTensor) FP32AddScalar(x float32, w, y []float32) {
	for i := range w {
		y[i] = x + w[i]
	}
}

func (*GoTensor) FP32AddVector(x, w, y []float32) {
	for i := range x {
		y[i] = x[i] + w[i]
	}
}
