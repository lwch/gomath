package tensor

import (
	"github.com/lwch/gomath"
	"github.com/lwch/gomath/internal/half"
)

func build(rows int64) (int64, []float64) {
	var cols int64
	if gomath.HasAVX512() {
		cols = 16 + 8
	} else if gomath.HasAVX() {
		cols = 8 + 4
	} else {
		cols = 3
	}
	x := make([]float64, rows*cols)
	for i := range x {
		x[i] = float64(i) + 1
	}
	output := make([]float64, rows*rows)
	for m := int64(0); m < rows; m++ {
		for n := int64(0); n < rows; n++ {
			var dx float64
			for d := int64(0); d < cols; d++ {
				dx += x[m*cols+d] * x[n*cols+d]
			}
			output[m*rows+n] = dx
		}
	}
	return cols, output
}

func equalF16(data []uint16, expect []float64) bool {
	if len(data) != len(expect) {
		return false
	}
	for i, v := range data {
		if v != half.Encode(float32(expect[i])) {
			return false
		}
	}
	return true
}

func equal[T float32 | float64](data []T, expect []float64) bool {
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
