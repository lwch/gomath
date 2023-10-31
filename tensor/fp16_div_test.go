package tensor

import (
	"testing"
)

func TestFP16Div(t *testing.T) {
	useCTensor(func() {
		testDiv(t, buildFP16Scalar, buildFP16Vector, buildFP16Matrix)
	})
}

func TestFP16DivGo(t *testing.T) {
	useGoTensor(func() {
		testDiv(t, buildFP16Scalar, buildFP16Vector, buildFP16Matrix)
	})
}

func BenchmarkFP16Div(b *testing.B) {
	useCTensor(func() {
		x := buildFP16Matrix(benchmarkRows, benchmarkCols)
		for i := 0; i < b.N; i++ {
			x.Div(x)
		}
	})
}

func BenchmarkFP16DivGo(b *testing.B) {
	useGoTensor(func() {
		x := buildFP16Matrix(benchmarkRows, benchmarkCols)
		for i := 0; i < b.N; i++ {
			x.Div(x)
		}
	})
}

func BenchmarkFP16DivScalar(b *testing.B) {
	useCTensor(func() {
		scalar := getScalar()
		x := buildFP16Matrix(benchmarkRows, benchmarkCols)
		w := NewFloat16([]float32{scalar}, []int64{1})
		for i := 0; i < b.N; i++ {
			x.Div(w)
		}
	})
}

func BenchmarkFP16DivScalarGo(b *testing.B) {
	useGoTensor(func() {
		scalar := getScalar()
		x := buildFP16Matrix(benchmarkRows, benchmarkCols)
		w := NewFloat16([]float32{scalar}, []int64{1})
		for i := 0; i < b.N; i++ {
			x.Div(w)
		}
	})
}

func BenchmarkFP16DivVector(b *testing.B) {
	useCTensor(func() {
		x := buildFP16Matrix(benchmarkRows, benchmarkCols)
		w := buildFP16Vector(benchmarkCols)
		for i := 0; i < b.N; i++ {
			x.Div(w)
		}
	})
}

func BenchmarkFP16DivVectorGo(b *testing.B) {
	useGoTensor(func() {
		x := buildFP16Matrix(benchmarkRows, benchmarkCols)
		w := buildFP16Vector(benchmarkCols)
		for i := 0; i < b.N; i++ {
			x.Div(w)
		}
	})
}
