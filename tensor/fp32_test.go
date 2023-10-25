package tensor

import (
	"testing"

	"github.com/lwch/gomath"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
)

const testRows = 8

func buildFP32(rows, cols int64) gomath.Tensor {
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = float32(i) + 1
	}
	return NewFloat32(data, []int64{rows, cols})
}

func TestFP32MatMul(t *testing.T) {
	debug = false
	cols := getCols()
	expect := computeMatMul(testRows, cols)
	x := buildFP32(testRows, cols)
	result := x.MatMul(x).(*Float32).data
	if !equal(result, expect) {
		t.Fatalf("expect %v, got %v", expect, result)
	}
}

func BenchmarkFP32MatMul(b *testing.B) {
	debug = false
	x := buildFP32(64, 4096)
	for i := 0; i < b.N; i++ {
		x.MatMul(x)
	}
}

func TestFP32MatMulGo(t *testing.T) {
	debug = true
	cols := getCols()
	expect := computeMatMul(testRows, cols)
	x := buildFP32(testRows, cols)
	result := x.MatMul(x).(*Float32).data
	if !equal(result, expect) {
		t.Fatalf("expect %v, got %v", expect, result)
	}
}

func BenchmarkFP32MatMulGo(b *testing.B) {
	debug = true
	x := buildFP32(64, 4096)
	for i := 0; i < b.N; i++ {
		x.MatMul(x)
	}
}

func fp32GoNumMatMul(rows, cols int64) gomath.Tensor {
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = float32(i) + 1
	}
	x := blas32.General{
		Rows:   int(rows),
		Cols:   int(cols),
		Stride: int(cols),
		Data:   data,
	}
	y := blas32.General{
		Rows:   int(rows),
		Cols:   int(cols),
		Stride: int(cols),
		Data:   data,
	}
	z := blas32.General{
		Rows:   int(rows),
		Cols:   int(rows),
		Stride: int(rows),
		Data:   make([]float32, rows*rows),
	}
	blas32.Gemm(blas.NoTrans, blas.Trans, 1, x, y, 0, z)
	return NewFloat32(z.Data, []int64{rows, rows})
}

func BenchmarkFP32MatMulGoNum(b *testing.B) {
	for i := 0; i < b.N; i++ {
		fp32GoNumMatMul(64, 4096)
	}
}
