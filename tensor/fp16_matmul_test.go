package tensor

import (
	"testing"

	"github.com/lwch/gomath"
)

func buildFP16Vector(cols int64) gomath.Tensor {
	data := make([]float32, cols)
	for i := range data {
		data[i] = float32(i) + 1
	}
	return NewFloat16(data, []int64{cols})
}

func buildFP16Matrix(rows, cols int64) gomath.Tensor {
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = float32(i) + 1
	}
	return NewFloat16(data, []int64{rows, cols})
}

func TestFP16MatMul(t *testing.T) {
	useCTensor(func() {
		testMatMul(t, buildFP16Vector, buildFP16Matrix)
	})
}

func BenchmarkFP16MatMul(b *testing.B) {
	useCTensor(func() {
		x := buildFP16Matrix(benchmarkRows, benchmarkCols)
		for i := 0; i < b.N; i++ {
			x.MatMul(x)
		}
	})
}

func TestFP16MatMulGo(t *testing.T) {
	useGoTensor(func() {
		testMatMul(t, buildFP16Vector, buildFP16Matrix)
	})
}

func BenchmarkFP16MatMulGo(b *testing.B) {
	useGoTensor(func() {
		x := buildFP16Matrix(benchmarkRows, benchmarkCols)
		for i := 0; i < b.N; i++ {
			x.MatMul(x)
		}
	})
}
