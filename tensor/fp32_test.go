package tensor

import (
	"testing"

	"github.com/lwch/gomath"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
)

func fp32MatMul(rows, cols int64) gomath.Tensor {
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = float32(i) + 1
	}
	x := NewFloat32(data, []int64{rows, cols})
	y := NewFloat32(data, []int64{rows, cols})
	return x.MatMul(y)
}

func TestFP32MatMul(t *testing.T) {
	debug = false
	cols, expect := build(2)
	ts := fp32MatMul(2, cols)
	result := ts.(*Float32).data
	if !equal(result, expect) {
		t.Fatalf("expect %v, got %v", expect, result)
	}
}

func BenchmarkFP32MatMul(b *testing.B) {
	debug = false
	for i := 0; i < b.N; i++ {
		fp32MatMul(64, 4096)
	}
}

func TestFP32MatMulGO(t *testing.T) {
	debug = true
	cols, expect := build(2)
	ts := fp32MatMul(2, cols)
	result := ts.(*Float32).data
	if !equal(result, expect) {
		t.Fatalf("expect %v, got %v", expect, result)
	}
}

func BenchmarkFP32MatMulGo(b *testing.B) {
	debug = true
	for i := 0; i < b.N; i++ {
		fp32MatMul(64, 4096)
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
