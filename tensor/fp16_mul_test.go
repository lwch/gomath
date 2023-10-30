package tensor

import (
	"testing"

	"github.com/lwch/gomath"
	"github.com/lwch/gomath/internal/half"
)

func buildFP16Scalar(n float32) gomath.Tensor {
	return NewFloat16([]float32{n}, []int64{1})
}

func TestFP16Mul(t *testing.T) {
	useCTensor(func() {
		testMul(t, buildFP16Matrix, buildFP16Scalar)
	})
}

func TestFP16MulGo(t *testing.T) {
	useGoTensor(func() {
		testMul(t, buildFP16Matrix, buildFP16Scalar)
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

func TestFP16MulVector(t *testing.T) {
	useCTensor(func() {
		rows := getRows()
		cols := getCols()
		expect, vec := computeMulVector(rows, cols)
		x := buildFP16Matrix(rows, cols)
		result := x.Mul(NewFloat16(vec, []int64{cols})).Storage()
		if !equal(result, expect) {
			tmp := make([]float32, result.Size())
			half.DecodeArray(result.Data().([]uint16), tmp)
			t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, tmp)
		}
	})
}

func TestFP16MulVectorGo(t *testing.T) {
	useGoTensor(func() {
		rows := getRows()
		cols := getCols()
		expect, vec := computeMulVector(rows, cols)
		x := buildFP16Matrix(rows, cols)
		result := x.Mul(NewFloat16(vec, []int64{cols})).Storage()
		if !equal(result, expect) {
			tmp := make([]float32, result.Size())
			half.DecodeArray(result.Data().([]uint16), tmp)
			t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, tmp)
		}
	})
}

func BenchmarkFP16MulVector(b *testing.B) {
	useCTensor(func() {
		_, vec := computeMulVector(benchmarkRows, benchmarkCols)
		x := buildFP16Matrix(benchmarkRows, benchmarkCols)
		w := NewFloat16(vec, []int64{benchmarkCols})
		for i := 0; i < b.N; i++ {
			x.Mul(w)
		}
	})
}

func BenchmarkFP16MulVectorGo(b *testing.B) {
	useGoTensor(func() {
		_, vec := computeMulVector(benchmarkRows, benchmarkCols)
		x := buildFP16Matrix(benchmarkRows, benchmarkCols)
		w := NewFloat16(vec, []int64{benchmarkCols})
		for i := 0; i < b.N; i++ {
			x.Mul(w)
		}
	})
}

func TestFP16Div(t *testing.T) {
	useCTensor(func() {
		rows := getRows()
		cols := getCols()
		expect := computeDiv(rows, cols)
		x := buildFP16Matrix(rows, cols)
		result := x.Div(x).Storage()
		if !equal(result, expect) {
			tmp := make([]float32, result.Size())
			half.DecodeArray(result.Data().([]uint16), tmp)
			t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, tmp)
		}
	})
}

func TestFP16DivGo(t *testing.T) {
	useGoTensor(func() {
		rows := getRows()
		cols := getCols()
		expect := computeDiv(rows, cols)
		x := buildFP16Matrix(rows, cols)
		result := x.Div(x).Storage()
		if !equal(result, expect) {
			tmp := make([]float32, result.Size())
			half.DecodeArray(result.Data().([]uint16), tmp)
			t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, tmp)
		}
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

func TestFP16DivScalar(t *testing.T) {
	useCTensor(func() {
		scalar := getScalar()
		rows := getRows()
		cols := getCols()
		expect := computeDivScalar(rows, cols, scalar)
		x := buildFP16Matrix(rows, cols)
		result := x.Div(NewFloat16([]float32{scalar}, []int64{1})).Storage()
		if !equal(result, expect) {
			tmp := make([]float32, result.Size())
			half.DecodeArray(result.Data().([]uint16), tmp)
			t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, tmp)
		}
	})
}

func TestFP16DivScalarGo(t *testing.T) {
	useGoTensor(func() {
		scalar := getScalar()
		rows := getRows()
		cols := getCols()
		expect := computeDivScalar(rows, cols, scalar)
		x := buildFP16Matrix(rows, cols)
		result := x.Div(NewFloat16([]float32{scalar}, []int64{1})).Storage()
		if !equal(result, expect) {
			tmp := make([]float32, result.Size())
			half.DecodeArray(result.Data().([]uint16), tmp)
			t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, tmp)
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

func TestFP16DivVector(t *testing.T) {
	useCTensor(func() {
		rows := getRows()
		cols := getCols()
		expect, vec := computeDivVector(rows, cols)
		x := buildFP16Matrix(rows, cols)
		result := x.Div(NewFloat16(vec, []int64{cols})).Storage()
		if !equal(result, expect) {
			tmp := make([]float32, result.Size())
			half.DecodeArray(result.Data().([]uint16), tmp)
			t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, tmp)
		}
	})
}

func TestFP16DivVectorGo(t *testing.T) {
	useGoTensor(func() {
		rows := getRows()
		cols := getCols()
		expect, vec := computeDivVector(rows, cols)
		x := buildFP16Matrix(rows, cols)
		result := x.Div(NewFloat16(vec, []int64{cols})).Storage()
		if !equal(result, expect) {
			tmp := make([]float32, result.Size())
			half.DecodeArray(result.Data().([]uint16), tmp)
			t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, tmp)
		}
	})
}

func BenchmarkFP16DivVector(b *testing.B) {
	useCTensor(func() {
		_, vec := computeDivVector(benchmarkRows, benchmarkCols)
		x := buildFP16Matrix(benchmarkRows, benchmarkCols)
		w := NewFloat16(vec, []int64{benchmarkCols})
		for i := 0; i < b.N; i++ {
			x.Div(w)
		}
	})
}

func BenchmarkFP16DivVectorGo(b *testing.B) {
	useGoTensor(func() {
		_, vec := computeDivVector(benchmarkRows, benchmarkCols)
		x := buildFP16Matrix(benchmarkRows, benchmarkCols)
		w := NewFloat16(vec, []int64{benchmarkCols})
		for i := 0; i < b.N; i++ {
			x.Div(w)
		}
	})
}
