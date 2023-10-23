package tensor

import (
	"testing"

	"github.com/lwch/gomath"
)

func f32MatMul(rows, cols int64) gomath.Tensor {
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = float32(i) + 1
	}
	x := NewFloat32(data, []int64{rows, cols})
	y := NewFloat32(data, []int64{rows, cols})
	return x.MatMul(y)
}

func TestF32MatMul(t *testing.T) {
	ts := f32MatMul(2, 3)
	result := ts.(*Float32).data
	if result[0] != 14 || result[1] != 32 || result[2] != 32 || result[3] != 77 {
		t.Fatal(result)
	}
}

func BenchmarkF32MatMul(b *testing.B) {
	for i := 0; i < b.N; i++ {
		f32MatMul(64, 4096)
	}
}

func TestF32MatMulGO(t *testing.T) {
	debug = true
	ts := f32MatMul(2, 3)
	result := ts.(*Float32).data
	if result[0] != 14 || result[1] != 32 || result[2] != 32 || result[3] != 77 {
		t.Fatal(result)
	}
}

func BenchmarkF32MatMulGO(b *testing.B) {
	debug = true
	for i := 0; i < b.N; i++ {
		f32MatMul(64, 4096)
	}
}
