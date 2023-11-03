package tensor

type TensorImpl interface {
	// fp16
	FP16Dot(x, w []uint16) uint16
	// mul
	FP16Mul(x, w, y []uint16)
	FP16ScalarMul(x uint16, w, y []uint16)
	// div
	FP16Div(x, w, y []uint16)
	FP16ScalarDiv(x uint16, w, y []uint16)
	// add
	FP16Add(x, w, y []uint16)
	FP16ScalarAdd(x uint16, w, y []uint16)
	// sub
	FP16Sub(x, w, y []uint16)
	FP16ScalarSub(x uint16, w, y []uint16)
	// pow
	FP16Pow(x []uint16, n uint16, y []uint16)

	// fp32
	FP32Dot(x, w []float32) float32
	// mul
	FP32Mul(x, w, y []float32)
	FP32ScalarMul(x float32, w, y []float32)
	// div
	FP32Div(x, w, y []float32)
	FP32ScalarDiv(x float32, w, y []float32)
	// add
	FP32Add(x, w, y []float32)
	FP32ScalarAdd(x float32, w, y []float32)
	// sub
	FP32Sub(x, w, y []float32)
	FP32ScalarSub(x float32, w, y []float32)
	// pow
	FP32Pow(x []float32, n float32, y []float32)
}
