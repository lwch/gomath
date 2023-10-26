package tensor

import (
	"math"
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

func computeMulVector(rows, cols int64) ([]float32, []float32) {
	x := make([]float32, rows*cols)
	w := make([]float32, cols)
	for i := range x {
		x[i] = float32(i) + 1
	}
	for i := range w {
		w[i] = float32(i) + 1
	}
	output := make([]float32, rows*cols)
	for i := int64(0); i < rows*cols; i++ {
		output[i] = x[i] * w[i%cols]
	}
	return output, w
}

func computeDiv(rows, cols int64) []float32 {
	x := make([]float32, rows*cols)
	for i := range x {
		x[i] = float32(i) + 1
	}
	output := make([]float32, rows*cols)
	for i := int64(0); i < rows*cols; i++ {
		output[i] = x[i] / x[i]
	}
	return output
}

func computeDivScalar(rows, cols int64, scalar float32) []float32 {
	x := make([]float32, rows*cols)
	for i := range x {
		x[i] = float32(i) + 1
	}
	output := make([]float32, rows*cols)
	for i := int64(0); i < rows*cols; i++ {
		output[i] = x[i] / scalar
	}
	return output
}

func computeDivVector(rows, cols int64) ([]float32, []float32) {
	x := make([]float32, rows*cols)
	w := make([]float32, cols)
	for i := range x {
		x[i] = float32(i) + 1
	}
	for i := range w {
		w[i] = float32(i) + 1
	}
	output := make([]float32, rows*cols)
	for i := int64(0); i < rows*cols; i++ {
		output[i] = x[i] / w[i%cols]
	}
	return output, w
}

func equalFP16(data []uint16, expect []float32) bool {
	if len(data) != len(expect) {
		return false
	}
	ok := true
	var std float32
	for i, v := range data {
		std += float32(math.Pow(float64(half.Decode(v)-expect[i]), 2))
		if v != half.Encode(expect[i]) {
			ok = false
		}
	}
	if ok {
		return true
	}
	std /= float32(len(data))
	return std < 0.01
}

func equal(data, expect []float32) bool {
	if len(data) != len(expect) {
		return false
	}
	ok := true
	var std float32
	for i, v := range data {
		std += float32(math.Pow(float64(v-expect[i]), 2))
		if v != expect[i] {
			ok = false
		}
	}
	if ok {
		return true
	}
	std /= float32(len(data))
	return std < 0.01
}
