package tensor

type TensorImpl interface {
	// fp16
	FP16DotVector(x, w []uint16) uint16
	// mul
	FP16MulScalar(x uint16, w, y []uint16)
	FP16MulVector(x, w, y []uint16)
	// div
	FP16DivScalar(x uint16, w, y []uint16)
	FP16DivVector(x, w, y []uint16)
	// add
	FP16AddScalar(x uint16, w, y []uint16)
	FP16AddVector(x, w, y []uint16)

	// fp32
	FP32DotVector(x, w []float32) float32
	// mul
	FP32MulScalar(x float32, w, y []float32)
	FP32MulVector(x, w, y []float32)
	// div
	FP32DivScalar(x float32, w, y []float32)
	FP32DivVector(x, w, y []float32)
	// add
	FP32AddScalar(x float32, w, y []float32)
	FP32AddVector(x, w, y []float32)
}
