package tensor

import (
	"testing"

	"github.com/lwch/gomath/internal/half"
)

func TestFP16Mul(t *testing.T) {
	debug = false
	rows := getRows()
	cols := getCols()
	expect := computeMul(rows, cols)
	x := buildFP16(rows, cols)
	result := x.Mul(x).(*Float16).data
	if !equalF16(result, expect) {
		tmp := make([]float32, len(result))
		half.DecodeArray(result, tmp)
		t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, tmp)
	}
}

func TestFP16MulGo(t *testing.T) {
	debug = true
	rows := getRows()
	cols := getCols()
	expect := computeMul(rows, cols)
	x := buildFP16(rows, cols)
	result := x.Mul(x).(*Float16).data
	if !equalF16(result, expect) {
		tmp := make([]float32, len(result))
		half.DecodeArray(result, tmp)
		t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, tmp)
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

func TestFP16MulScalar(t *testing.T) {
	debug = false
	scalar := getScalar()
	rows := getRows()
	cols := getCols()
	expect := computeMulScalar(rows, cols, scalar)
	x := buildFP16(rows, cols)
	result := x.Mul(NewFloat16([]float32{scalar}, []int64{1})).(*Float16).data
	if !equalF16(result, expect) {
		tmp := make([]float32, len(result))
		half.DecodeArray(result, tmp)
		t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, tmp)
	}
}

func TestFP16MulScalarGo(t *testing.T) {
	debug = true
	scalar := getScalar()
	rows := getRows()
	cols := getCols()
	expect := computeMulScalar(rows, cols, scalar)
	x := buildFP16(rows, cols)
	result := x.Mul(NewFloat16([]float32{scalar}, []int64{1})).(*Float16).data
	if !equalF16(result, expect) {
		tmp := make([]float32, len(result))
		half.DecodeArray(result, tmp)
		t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, tmp)
	}
}

func BenchmarkFP16MulScalar(b *testing.B) {
	debug = false
	scalar := getScalar()
	x := buildFP16(64, 4096)
	w := NewFloat16([]float32{scalar}, []int64{1})
	for i := 0; i < b.N; i++ {
		x.Mul(w)
	}
}

func BenchmarkFP16MulScalarGo(b *testing.B) {
	debug = true
	scalar := getScalar()
	x := buildFP16(64, 4096)
	w := NewFloat16([]float32{scalar}, []int64{1})
	for i := 0; i < b.N; i++ {
		x.Mul(w)
	}
}
