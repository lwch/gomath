package tensor

import (
	"testing"

	"github.com/lwch/gomath"
)

func buildFP16Scalar(n float32) gomath.Tensor {
	return NewFloat16([]float32{n}, []int64{1})
}

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
	testC(t, func(t *testing.T) {
		testMatMul(t, buildFP16Vector, buildFP16Matrix)
	})
	testGo(t, func(t *testing.T) {
		testMatMul(t, buildFP16Vector, buildFP16Matrix)
	})
}

func BenchmarkFP16MatMul(b *testing.B) {
	benchmarkMatMul(b, false, buildFP16Vector, buildFP16Matrix,
		func(x, w gomath.Tensor) {
			x.MatMul(w)
		})
}

func TestFP16Mul(t *testing.T) {
	testC(t, func(t *testing.T) {
		testMul(t, buildFP16Scalar, buildFP16Vector, buildFP16Matrix)
	})
	testGo(t, func(t *testing.T) {
		testMul(t, buildFP16Scalar, buildFP16Vector, buildFP16Matrix)
	})
}

func BenchmarkFP16Mul(b *testing.B) {
	batchBenchmark(b, buildFP16Scalar, buildFP16Vector, buildFP16Matrix,
		func(x, w gomath.Tensor) {
			x.Mul(w)
		})
}

func TestFP16Div(t *testing.T) {
	testC(t, func(t *testing.T) {
		testDiv(t, buildFP16Scalar, buildFP16Vector, buildFP16Matrix)
	})
	testGo(t, func(t *testing.T) {
		testDiv(t, buildFP16Scalar, buildFP16Vector, buildFP16Matrix)
	})
}

func BenchmarkFP16Div(b *testing.B) {
	batchBenchmark(b, buildFP16Scalar, buildFP16Vector, buildFP16Matrix,
		func(x, w gomath.Tensor) {
			x.Div(w)
		})
}
