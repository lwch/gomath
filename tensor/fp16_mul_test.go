package tensor

import (
	"testing"

	"github.com/lwch/gomath"
)

func buildFP16Scalar(n float32) gomath.Tensor {
	return NewFloat16([]float32{n}, []int64{1})
}

func TestFP16Mul(t *testing.T) {
	useCTensor(func() {
		testMul(t, buildFP16Scalar, buildFP16Vector, buildFP16Matrix)
	})
}

func TestFP16MulGo(t *testing.T) {
	useGoTensor(func() {
		testMul(t, buildFP16Scalar, buildFP16Vector, buildFP16Matrix)
	})
}

func BenchmarkFP16Mul(b *testing.B) {
	useCTensor(func() {
		x := buildFP16Matrix(benchmarkRows, benchmarkCols)
		for i := 0; i < b.N; i++ {
			x.Mul(x)
		}
	})
}

func BenchmarkFP16MulGo(b *testing.B) {
	useGoTensor(func() {
		x := buildFP16Matrix(benchmarkRows, benchmarkCols)
		for i := 0; i < b.N; i++ {
			x.Mul(x)
		}
	})
}

func BenchmarkFP16MulScalar(b *testing.B) {
	useCTensor(func() {
		scalar := getScalar()
		x := buildFP16Matrix(benchmarkRows, benchmarkCols)
		w := NewFloat16([]float32{scalar}, []int64{1})
		for i := 0; i < b.N; i++ {
			x.Mul(w)
		}
	})
}

func BenchmarkFP16MulScalarGo(b *testing.B) {
	useGoTensor(func() {
		scalar := getScalar()
		x := buildFP16Matrix(benchmarkRows, benchmarkCols)
		w := NewFloat16([]float32{scalar}, []int64{1})
		for i := 0; i < b.N; i++ {
			x.Mul(w)
		}
	})
}

func BenchmarkFP16MulVector(b *testing.B) {
	useCTensor(func() {
		x := buildFP16Matrix(benchmarkRows, benchmarkCols)
		w := buildFP16Vector(benchmarkCols)
		for i := 0; i < b.N; i++ {
			x.Mul(w)
		}
	})
}

func BenchmarkFP16MulVectorGo(b *testing.B) {
	useGoTensor(func() {
		x := buildFP16Matrix(benchmarkRows, benchmarkCols)
		w := buildFP16Vector(benchmarkCols)
		for i := 0; i < b.N; i++ {
			x.Mul(w)
		}
	})
}
