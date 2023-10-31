package tensor

import "testing"

func TestFP32Div(t *testing.T) {
	useCTensor(func() {
		testDiv(t, buildFP32Scalar, buildFP32Vector, buildFP32Matrix)
	})
}

func TestFP32DivGo(t *testing.T) {
	useGoTensor(func() {
		testDiv(t, buildFP32Scalar, buildFP32Vector, buildFP32Matrix)
	})
}

func BenchmarkFP32Div(b *testing.B) {
	useCTensor(func() {
		x := buildFP32Matrix(benchmarkRows, benchmarkCols)
		for i := 0; i < b.N; i++ {
			x.Div(x)
		}
	})
}

func BenchmarkFP32DivGo(b *testing.B) {
	useGoTensor(func() {
		x := buildFP32Matrix(benchmarkRows, benchmarkCols)
		for i := 0; i < b.N; i++ {
			x.Div(x)
		}
	})
}

func BenchmarkFP32DivScalar(b *testing.B) {
	useCTensor(func() {
		scalar := getScalar()
		x := buildFP32Matrix(benchmarkRows, benchmarkCols)
		w := NewFloat32([]float32{scalar}, []int64{1})
		for i := 0; i < b.N; i++ {
			x.Div(w)
		}
	})
}

func BenchmarkFP32DivScalarGo(b *testing.B) {
	useGoTensor(func() {
		scalar := getScalar()
		x := buildFP32Matrix(benchmarkRows, benchmarkCols)
		w := NewFloat32([]float32{scalar}, []int64{1})
		for i := 0; i < b.N; i++ {
			x.Div(w)
		}
	})
}

func BenchmarkFP32DivVector(b *testing.B) {
	useCTensor(func() {
		x := buildFP32Matrix(benchmarkRows, benchmarkCols)
		w := buildFP32Vector(benchmarkCols)
		for i := 0; i < b.N; i++ {
			x.Div(w)
		}
	})
}

func BenchmarkFP32DivVectorGo(b *testing.B) {
	useGoTensor(func() {
		x := buildFP32Matrix(benchmarkRows, benchmarkCols)
		w := buildFP32Vector(benchmarkCols)
		for i := 0; i < b.N; i++ {
			x.Div(w)
		}
	})
}
