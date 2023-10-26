package tensor

import (
	"math/rand"

	"github.com/lwch/gomath"
	"github.com/lwch/gomath/internal/half"
)

func getScalar() float32 {
	return float32(rand.Intn(10) + 1)
}

func getRows() int64 {
	return rand.Int63n(7) + 1
}

func getCols() int64 {
	var cols int64
	if gomath.HasAVX512() {
		cols = 16 + 8
	} else if gomath.HasAVX() {
		cols = 8 + 4
	} else {
		cols = 3
	}
	return cols
}

func computeMatMul(rows, cols int64) []float32 {
	x := make([]float32, rows*cols)
	for i := range x {
		x[i] = float32(i) + 1
	}
	output := make([]float32, rows*rows)
	for m := int64(0); m < rows; m++ {
		for n := int64(0); n < rows; n++ {
			var dx float32
			for d := int64(0); d < cols; d++ {
				dx += x[m*cols+d] * x[n*cols+d]
			}
			output[m*rows+n] = dx
		}
	}
	return output
}

func computeMul(rows, cols int64) []float32 {
	x := make([]float32, rows*cols)
	for i := range x {
		x[i] = float32(i) + 1
	}
	output := make([]float32, rows*cols)
	for i := int64(0); i < rows*cols; i++ {
		output[i] = x[i] * x[i]
	}
	return output
}

func computeMulScalar(rows, cols int64, scalar float32) []float32 {
	x := make([]float32, rows*cols)
	for i := range x {
		x[i] = float32(i) + 1
	}
	output := make([]float32, rows*cols)
	for i := int64(0); i < rows*cols; i++ {
		output[i] = x[i] * scalar
	}
	return output
}

func equalF16(data []uint16, expect []float32) bool {
	if len(data) != len(expect) {
		return false
	}
	for i, v := range data {
		if v != half.Encode(expect[i]) {
			return false
		}
	}
	return true
}

func equal[T float32](data []T, expect []float32) bool {
	if len(data) != len(expect) {
		return false
	}
	for i, v := range data {
		if v != T(expect[i]) {
			return false
		}
	}
	return true
}
