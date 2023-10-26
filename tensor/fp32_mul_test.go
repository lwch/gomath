package tensor

import (
	"testing"
)

func TestFP32Mul(t *testing.T) {
	debug = false
	rows := getRows()
	cols := getCols()
	expect := computeMul(rows, cols)
	x := buildFP32(rows, cols)
	result := x.Mul(x).(*Float32).data
	if !equal(result, expect) {
		t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, result)
	}
}

func TestFP32MulGo(t *testing.T) {
	debug = true
	rows := getRows()
	cols := getCols()
	expect := computeMul(rows, cols)
	x := buildFP32(rows, cols)
	result := x.Mul(x).(*Float32).data
	if !equal(result, expect) {
		t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, result)
	}
}

func BenchmarkFP32Mul(b *testing.B) {
	debug = false
	x := buildFP32(64, 4096)
	for i := 0; i < b.N; i++ {
		x.Mul(x)
	}
}

func BenchmarkFP32MulGo(b *testing.B) {
	debug = true
	x := buildFP32(64, 4096)
	for i := 0; i < b.N; i++ {
		x.Mul(x)
	}
}

func TestFP32MulScalar(t *testing.T) {
	debug = false
	scalar := getScalar()
	rows := getRows()
	cols := getCols()
	expect := computeMulScalar(rows, cols, scalar)
	x := buildFP32(rows, cols)
	result := x.Mul(NewFloat32([]float32{scalar}, []int64{1})).(*Float32).data
	if !equal(result, expect) {
		t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, result)
	}
}

func TestFP32MulScalarGo(t *testing.T) {
	debug = true
	scalar := getScalar()
	rows := getRows()
	cols := getCols()
	expect := computeMulScalar(rows, cols, scalar)
	x := buildFP32(rows, cols)
	result := x.Mul(NewFloat32([]float32{scalar}, []int64{1})).(*Float32).data
	if !equal(result, expect) {
		t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, result)
	}
}

func BenchmarkFP32MulScalar(b *testing.B) {
	debug = false
	scalar := getScalar()
	x := buildFP32(64, 4096)
	w := NewFloat32([]float32{scalar}, []int64{1})
	for i := 0; i < b.N; i++ {
		x.Mul(w)
	}
}

func BenchmarkFP32MulScalarGo(b *testing.B) {
	debug = true
	scalar := getScalar()
	x := buildFP32(64, 4096)
	w := NewFloat32([]float32{scalar}, []int64{1})
	for i := 0; i < b.N; i++ {
		x.Mul(w)
	}
}

func TestFP32MulVector(t *testing.T) {
	debug = false
	rows := getRows()
	cols := getCols()
	expect, vec := computeMulVector(rows, cols)
	x := buildFP32(rows, cols)
	result := x.Mul(NewFloat32(vec, []int64{cols})).(*Float32).data
	if !equal(result, expect) {
		t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, result)
	}
}

func TestFP32MulVectorGo(t *testing.T) {
	debug = true
	rows := getRows()
	cols := getCols()
	expect, vec := computeMulVector(rows, cols)
	x := buildFP32(rows, cols)
	result := x.Mul(NewFloat32(vec, []int64{cols})).(*Float32).data
	if !equal(result, expect) {
		t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, result)
	}
}

func BenchmarkFP32MulVector(b *testing.B) {
	debug = false
	_, vec := computeMulVector(64, 4096)
	x := buildFP32(64, 4096)
	w := NewFloat32(vec, []int64{4096})
	for i := 0; i < b.N; i++ {
		x.Mul(w)
	}
}

func BenchmarkFP32MulVectorGo(b *testing.B) {
	debug = true
	_, vec := computeMulVector(64, 4096)
	x := buildFP32(64, 4096)
	w := NewFloat32(vec, []int64{4096})
	for i := 0; i < b.N; i++ {
		x.Mul(w)
	}
}

func TestFP32Div(t *testing.T) {
	debug = false
	rows := getRows()
	cols := getCols()
	expect := computeDiv(rows, cols)
	x := buildFP32(rows, cols)
	result := x.Div(x).(*Float32).data
	if !equal(result, expect) {
		t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, result)
	}
}

func TestFP32DivGo(t *testing.T) {
	debug = true
	rows := getRows()
	cols := getCols()
	expect := computeDiv(rows, cols)
	x := buildFP32(rows, cols)
	result := x.Div(x).(*Float32).data
	if !equal(result, expect) {
		t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, result)
	}
}

func BenchmarkFP32Div(b *testing.B) {
	debug = false
	x := buildFP32(64, 4096)
	for i := 0; i < b.N; i++ {
		x.Div(x)
	}
}

func BenchmarkFP32DivGo(b *testing.B) {
	debug = true
	x := buildFP32(64, 4096)
	for i := 0; i < b.N; i++ {
		x.Div(x)
	}
}

func TestFP32DivScalar(t *testing.T) {
	debug = false
	scalar := getScalar()
	rows := getRows()
	cols := getCols()
	expect := computeDivScalar(rows, cols, scalar)
	x := buildFP32(rows, cols)
	result := x.Div(NewFloat32([]float32{scalar}, []int64{1})).(*Float32).data
	if !equal(result, expect) {
		t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, result)
	}
}

func TestFP32DivScalarGo(t *testing.T) {
	debug = true
	scalar := getScalar()
	rows := getRows()
	cols := getCols()
	expect := computeDivScalar(rows, cols, scalar)
	x := buildFP32(rows, cols)
	result := x.Div(NewFloat32([]float32{scalar}, []int64{1})).(*Float32).data
	if !equal(result, expect) {
		t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, result)
	}
}

func BenchmarkFP32DivScalar(b *testing.B) {
	debug = false
	scalar := getScalar()
	x := buildFP32(64, 4096)
	w := NewFloat32([]float32{scalar}, []int64{1})
	for i := 0; i < b.N; i++ {
		x.Div(w)
	}
}

func BenchmarkFP32DivScalarGo(b *testing.B) {
	debug = true
	scalar := getScalar()
	x := buildFP32(64, 4096)
	w := NewFloat32([]float32{scalar}, []int64{1})
	for i := 0; i < b.N; i++ {
		x.Div(w)
	}
}

func TestFP32DivVector(t *testing.T) {
	debug = false
	rows := getRows()
	cols := getCols()
	expect, vec := computeDivVector(rows, cols)
	x := buildFP32(rows, cols)
	result := x.Div(NewFloat32(vec, []int64{cols})).(*Float32).data
	if !equal(result, expect) {
		t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, result)
	}
}

func TestFP32DivVectorGo(t *testing.T) {
	debug = true
	rows := getRows()
	cols := getCols()
	expect, vec := computeDivVector(rows, cols)
	x := buildFP32(rows, cols)
	result := x.Div(NewFloat32(vec, []int64{cols})).(*Float32).data
	if !equal(result, expect) {
		t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, result)
	}
}

func BenchmarkFP32DivVector(b *testing.B) {
	debug = false
	_, vec := computeDivVector(64, 4096)
	x := buildFP32(64, 4096)
	w := NewFloat32(vec, []int64{4096})
	for i := 0; i < b.N; i++ {
		x.Div(w)
	}
}

func BenchmarkFP32DivVectorGo(b *testing.B) {
	debug = true
	_, vec := computeDivVector(64, 4096)
	x := buildFP32(64, 4096)
	w := NewFloat32(vec, []int64{4096})
	for i := 0; i < b.N; i++ {
		x.Div(w)
	}
}
