package tensor

type TensorImpl interface {
	// fp16
	FP16DotVector(x, w []uint16) uint16
	FP16MulVector(x, w, y []uint16)
	FP16MulScalar(x []uint16, w uint16, y []uint16)
	FP16DivVector(x, w, y []uint16)
	// fp32
	FP32DotVector(x, w []float32) float32
	FP32MulVector(x, w, y []float32)
	FP32MulScalar(x []float32, w float32, y []float32)
	FP32DivVector(x, w, y []float32)
}
