package tensor

import (
	"testing"

	"github.com/lwch/gomath"
	"github.com/lwch/gomath/internal/half"
)

func buildFP16(rows, cols int64) gomath.Tensor {
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = float32(i) + 1
	}
	return NewFloat16(data, []int64{rows, cols})
}

func TestFP16MatMul(t *testing.T) {
	debug = false
	cols := getCols()
	expect := computeMatMul(testRows, cols)
	x := buildFP16(testRows, cols)
	result := x.MatMul(x).(*Float16).data
	if !equalF16(result, expect) {
		tmp := make([]float32, len(result))
		half.DecodeArray(result, tmp)
		t.Fatalf("expect=%v, got=%v", expect, tmp)
	}
}

func BenchmarkFP16MatMul(b *testing.B) {
	debug = false
	x := buildFP16(64, 4096)
	for i := 0; i < b.N; i++ {
		x.MatMul(x)
	}
}

func TestFP16MatMulGo(t *testing.T) {
	debug = true
	cols := getCols()
	expect := computeMatMul(testRows, cols)
	x := buildFP16(testRows, cols)
	result := x.MatMul(x).(*Float16).data
	if !equalF16(result, expect) {
		tmp := make([]float32, len(result))
		half.DecodeArray(result, tmp)
		t.Fatalf("expect=%v, got=%v", expect, tmp)
	}
}

func BenchmarkFP16MatMulGo(b *testing.B) {
	debug = true
	x := buildFP16(64, 4096)
	for i := 0; i < b.N; i++ {
		x.MatMul(x)
	}
}

func TestFP16Mul(t *testing.T) {
	debug = false
	cols := getCols()
	expect := computeMul(testRows, cols)
	x := buildFP16(testRows, cols)
	result := x.Mul(x).(*Float16).data
	if !equalF16(result, expect) {
		tmp := make([]float32, len(result))
		half.DecodeArray(result, tmp)
		t.Fatalf("expect=%v, got=%v", expect, tmp)
	}
}

func TestFP16MulGo(t *testing.T) {
	debug = true
	cols := getCols()
	expect := computeMul(testRows, cols)
	x := buildFP16(testRows, cols)
	result := x.Mul(x).(*Float16).data
	if !equalF16(result, expect) {
		tmp := make([]float32, len(result))
		half.DecodeArray(result, tmp)
		t.Fatalf("expect=%v, got=%v", expect, tmp)
	}
}

func BenchmarkFP16Mul(b *testing.B) {
	debug = false
	x := buildFP16(64, 4096)
	for i := 0; i < b.N; i++ {
		x.Mul(x)
	}
}

func BenchmarkFP16MulGo(b *testing.B) {
	debug = true
	x := buildFP16(64, 4096)
	for i := 0; i < b.N; i++ {
		x.Mul(x)
	}
}
