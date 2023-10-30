package tensor

import (
	"testing"

	"github.com/lwch/gomath"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
)

func buildFP32(rows, cols int64) gomath.Tensor {
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = float32(i) + 1
	}
	return NewFloat32(data, []int64{rows, cols})
}

func TestFP32MatMul(t *testing.T) {
	useCTensor(func() {
		rows := getRows()
		cols := getCols()
		expect := computeMatMul(rows, cols)
		x := buildFP32(rows, cols)
		result := x.MatMul(x).Storage()
		if !equal(result, expect) {
			t.Fatalf("(%d, %d): expect %v, got %v", rows, cols, expect, result)
		}
	})
}

func BenchmarkFP32MatMul(b *testing.B) {
	useCTensor(func() {
		x := buildFP32(benchmarkRows, benchmarkCols)
		for i := 0; i < b.N; i++ {
			x.MatMul(x)
		}
	})
}

func TestFP32MatMulGo(t *testing.T) {
	useGoTensor(func() {
		rows := getRows()
		cols := getCols()
		expect := computeMatMul(rows, cols)
		x := buildFP32(rows, cols)
		result := x.MatMul(x).Storage()
		if !equal(result, expect) {
			t.Fatalf("(%d, %d): expect %v, got %v", rows, cols, expect, result)
		}
	})
}

func BenchmarkFP32MatMulGo(b *testing.B) {
	useGoTensor(func() {
		x := buildFP32(benchmarkRows, benchmarkCols)
		for i := 0; i < b.N; i++ {
			x.MatMul(x)
		}
	})
}

func fp32GoNumMatMul(x, y blas32.General, rows, cols int64) blas32.General {
	z := blas32.General{
		Rows:   int(rows),
		Cols:   int(rows),
		Stride: int(rows),
		Data:   make([]float32, rows*rows),
	}
	blas32.Gemm(blas.NoTrans, blas.Trans, 1, x, y, 0, z)
	return z
}

func BenchmarkFP32MatMulGoNum(b *testing.B) {
	x := blas32.General{
		Rows:   benchmarkRows,
		Cols:   benchmarkCols,
		Stride: benchmarkCols,
		Data:   make([]float32, benchmarkRows*benchmarkCols),
	}
	for i := range x.Data {
		x.Data[i] = float32(i) + 1
	}
	for i := 0; i < b.N; i++ {
		fp32GoNumMatMul(x, x, benchmarkRows, benchmarkCols)
	}
}
