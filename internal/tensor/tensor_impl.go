package tensor

type TensorImpl interface {
	// fp16
	FP16DotVector(x, w []uint16) uint16
	FP16Mul(x, w, y []uint16)
	// fp32
	FP32DotVector(x, w []float32) float32
	FP32Mul(x, w, y []float32)
}
