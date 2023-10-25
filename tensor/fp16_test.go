package tensor

import (
	"testing"

	"github.com/lwch/gomath"
	"github.com/lwch/gomath/internal/half"
)

func fp16MatMul(rows, cols int64) gomath.Tensor {
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = float32(i) + 1
	}
	x := NewFloat16(data, []int64{rows, cols})
	y := NewFloat16(data, []int64{rows, cols})
	return x.MatMul(y)
}

func TestFP16MatMul(t *testing.T) {
	debug = false
	cols, expect := build(2)
	ts := fp16MatMul(2, cols)
	result := ts.(*Float16).data
	if !equalF16(result, expect) {
		tmp := make([]float32, len(result))
		half.DecodeArray(result, tmp)
		t.Fatalf("expect=%v, got=%v", expect, tmp)
	}
}

func BenchmarkFP16MatMul(b *testing.B) {
	debug = false
	for i := 0; i < b.N; i++ {
		fp16MatMul(64, 4096)
	}
}

func TestFP16MatMulGO(t *testing.T) {
	debug = true
	cols, expect := build(2)
	ts := fp16MatMul(2, cols)
	result := ts.(*Float16).data
	if !equalF16(result, expect) {
		tmp := make([]float32, len(result))
		half.DecodeArray(result, tmp)
		t.Fatalf("expect=%v, got=%v", expect, tmp)
	}
}

func BenchmarkFP16MatMulGo(b *testing.B) {
	debug = true
	for i := 0; i < b.N; i++ {
		fp16MatMul(64, 4096)
	}
}
