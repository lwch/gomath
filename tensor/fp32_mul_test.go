package tensor

import (
	"testing"

	"github.com/lwch/gomath"
)

func buildFP32Scalar(n float32) gomath.Tensor {
	return NewFloat32([]float32{n}, []int64{1})
}

func TestFP32Mul(t *testing.T) {
	useCTensor(func() {
		testMul(t, buildFP32Scalar, buildFP32Vector, buildFP32Matrix)
	})
}

func TestFP32MulGo(t *testing.T) {
	useGoTensor(func() {
		testMul(t, buildFP32Scalar, buildFP32Vector, buildFP32Matrix)
	})
}

func BenchmarkFP32Mul(b *testing.B) {
	useCTensor(func() {
		x := buildFP32Matrix(benchmarkRows, benchmarkCols)
		for i := 0; i < b.N; i++ {
			x.Mul(x)
		}
	})
}

func BenchmarkFP32MulGo(b *testing.B) {
	useGoTensor(func() {
		x := buildFP32Matrix(benchmarkRows, benchmarkCols)
		for i := 0; i < b.N; i++ {
			x.Mul(x)
		}
	})
}

func BenchmarkFP32MulScalar(b *testing.B) {
	useCTensor(func() {
		scalar := getScalar()
		x := buildFP32Matrix(benchmarkRows, benchmarkCols)
		w := NewFloat32([]float32{scalar}, []int64{1})
		for i := 0; i < b.N; i++ {
			x.Mul(w)
		}
	})
}

func BenchmarkFP32MulScalarGo(b *testing.B) {
	useGoTensor(func() {
		scalar := getScalar()
		x := buildFP32Matrix(benchmarkRows, benchmarkCols)
		w := NewFloat32([]float32{scalar}, []int64{1})
		for i := 0; i < b.N; i++ {
			x.Mul(w)
		}
	})
}

func BenchmarkFP32MulVector(b *testing.B) {
	useCTensor(func() {
		x := buildFP32Matrix(benchmarkRows, benchmarkCols)
		w := buildFP32Vector(benchmarkCols)
		for i := 0; i < b.N; i++ {
			x.Mul(w)
		}
	})
}

func BenchmarkFP32MulVectorGo(b *testing.B) {
	useGoTensor(func() {
		x := buildFP32Matrix(benchmarkRows, benchmarkCols)
		w := buildFP32Vector(benchmarkCols)
		for i := 0; i < b.N; i++ {
			x.Mul(w)
		}
	})
}
