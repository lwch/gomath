package tensor

import (
	"testing"

	"github.com/lwch/gomath"
)

func buildFP32Scalar(n float32) gomath.Tensor {
	return NewFloat32([]float32{n}, []int64{1})
}

func buildFP32Vector(cols int64) gomath.Tensor {
	data := make([]float32, cols)
	for i := range data {
		data[i] = float32(i) + 1
	}
	return NewFloat32(data, []int64{cols})
}

func buildFP32Matrix(rows, cols int64) gomath.Tensor {
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = float32(i) + 1
	}
	return NewFloat32(data, []int64{rows, cols})
}

func TestFP32MatMul(t *testing.T) {
	testMatMul(t, buildFP32Vector, buildFP32Matrix)
}

func BenchmarkFP32MatMul(b *testing.B) {
	benchmarkMatMul(b, true, buildFP32Vector, buildFP32Matrix,
		func(x, w gomath.Tensor) {
			x.MatMul(w)
		})
}

func TestFP32Mul(t *testing.T) {
	testCompute(t, buildFP32Scalar, buildFP32Vector, buildFP32Matrix,
		func(x, w float32) float32 {
			return x * w
		}, func(x, w gomath.Tensor) gomath.Tensor {
			return x.Mul(w)
		})
}

func BenchmarkFP32Mul(b *testing.B) {
	benchmarkCompute(b, buildFP32Scalar, buildFP32Vector, buildFP32Matrix,
		func(x, w gomath.Tensor) {
			x.Mul(w)
		})
}

func TestFP32Div(t *testing.T) {
	testCompute(t, buildFP32Scalar, buildFP32Vector, buildFP32Matrix,
		func(x, w float32) float32 {
			return x / w
		}, func(x, w gomath.Tensor) gomath.Tensor {
			return x.Div(w)
		})
}

func BenchmarkFP32Div(b *testing.B) {
	benchmarkCompute(b, buildFP32Scalar, buildFP32Vector, buildFP32Matrix,
		func(x, w gomath.Tensor) {
			x.Div(w)
		})
}

func TestFP32Add(t *testing.T) {
	testCompute(t, buildFP32Scalar, buildFP32Vector, buildFP32Matrix,
		func(x, w float32) float32 {
			return x + w
		}, func(x, w gomath.Tensor) gomath.Tensor {
			return x.Add(w)
		})
}

func BenchmarkFP32Add(b *testing.B) {
	benchmarkCompute(b, buildFP32Scalar, buildFP32Vector, buildFP32Matrix,
		func(x, w gomath.Tensor) {
			x.Add(w)
		})
}

func TestFP32Sub(t *testing.T) {
	testCompute(t, buildFP32Scalar, buildFP32Vector, buildFP32Matrix,
		func(x, w float32) float32 {
			return x - w
		}, func(x, w gomath.Tensor) gomath.Tensor {
			return x.Sub(w)
		})
}

func BenchmarkFP32Sub(b *testing.B) {
	benchmarkCompute(b, buildFP32Scalar, buildFP32Vector, buildFP32Matrix,
		func(x, w gomath.Tensor) {
			x.Sub(w)
		})
}
