package utils

func ComputeStride(shape []int64) []int64 {
	strides := make([]int64, len(shape))
	strides[len(shape)-1] = 1
	for i := len(shape) - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * shape[i+1]
	}
	return strides
}
