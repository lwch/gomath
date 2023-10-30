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
	result := x.Mul(x).Storage()
	if !equal(result, expect) {
		t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, result.Data())
	}
}

func TestFP32MulGo(t *testing.T) {
	debug = true
	rows := getRows()
	cols := getCols()
	expect := computeMul(rows, cols)
	x := buildFP32(rows, cols)
	result := x.Mul(x).Storage()
	if !equal(result, expect) {
		t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, result.Data())
	}
}

func BenchmarkFP32Mul(b *testing.B) {
	debug = false
	x := buildFP32(benchmarkRows, benchmarkCols)
	for i := 0; i < b.N; i++ {
		x.Mul(x)
	}
}

func BenchmarkFP32MulGo(b *testing.B) {
	debug = true
	x := buildFP32(benchmarkRows, benchmarkCols)
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
	result := x.Mul(NewFloat32([]float32{scalar}, []int64{1})).Storage()
	if !equal(result, expect) {
		t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, result.Data())
	}
}

func TestFP32MulScalarGo(t *testing.T) {
	debug = true
	scalar := getScalar()
	rows := getRows()
	cols := getCols()
	expect := computeMulScalar(rows, cols, scalar)
	x := buildFP32(rows, cols)
	result := x.Mul(NewFloat32([]float32{scalar}, []int64{1})).Storage()
	if !equal(result, expect) {
		t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, result.Data())
	}
}

func BenchmarkFP32MulScalar(b *testing.B) {
	debug = false
	scalar := getScalar()
	x := buildFP32(benchmarkRows, benchmarkCols)
	w := NewFloat32([]float32{scalar}, []int64{1})
	for i := 0; i < b.N; i++ {
		x.Mul(w)
	}
}

func BenchmarkFP32MulScalarGo(b *testing.B) {
	debug = true
	scalar := getScalar()
	x := buildFP32(benchmarkRows, benchmarkCols)
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
	result := x.Mul(NewFloat32(vec, []int64{cols})).Storage()
	if !equal(result, expect) {
		t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, result.Data())
	}
}

func TestFP32MulVectorGo(t *testing.T) {
	debug = true
	rows := getRows()
	cols := getCols()
	expect, vec := computeMulVector(rows, cols)
	x := buildFP32(rows, cols)
	result := x.Mul(NewFloat32(vec, []int64{cols})).Storage()
	if !equal(result, expect) {
		t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, result.Data())
	}
}

func BenchmarkFP32MulVector(b *testing.B) {
	debug = false
	_, vec := computeMulVector(benchmarkRows, benchmarkCols)
	x := buildFP32(benchmarkRows, benchmarkCols)
	w := NewFloat32(vec, []int64{benchmarkCols})
	for i := 0; i < b.N; i++ {
		x.Mul(w)
	}
}

func BenchmarkFP32MulVectorGo(b *testing.B) {
	debug = true
	_, vec := computeMulVector(benchmarkRows, benchmarkCols)
	x := buildFP32(benchmarkRows, benchmarkCols)
	w := NewFloat32(vec, []int64{benchmarkCols})
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
	result := x.Div(x).Storage()
	if !equal(result, expect) {
		t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, result.Data())
	}
}

func TestFP32DivGo(t *testing.T) {
	debug = true
	rows := getRows()
	cols := getCols()
	expect := computeDiv(rows, cols)
	x := buildFP32(rows, cols)
	result := x.Div(x).Storage()
	if !equal(result, expect) {
		t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, result.Data())
	}
}

func BenchmarkFP32Div(b *testing.B) {
	debug = false
	x := buildFP32(benchmarkRows, benchmarkCols)
	for i := 0; i < b.N; i++ {
		x.Div(x)
	}
}

func BenchmarkFP32DivGo(b *testing.B) {
	debug = true
	x := buildFP32(benchmarkRows, benchmarkCols)
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
	result := x.Div(NewFloat32([]float32{scalar}, []int64{1})).Storage()
	if !equal(result, expect) {
		t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, result.Data())
	}
}

func TestFP32DivScalarGo(t *testing.T) {
	debug = true
	scalar := getScalar()
	rows := getRows()
	cols := getCols()
	expect := computeDivScalar(rows, cols, scalar)
	x := buildFP32(rows, cols)
	result := x.Div(NewFloat32([]float32{scalar}, []int64{1})).Storage()
	if !equal(result, expect) {
		t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, result.Data())
	}
}

func BenchmarkFP32DivScalar(b *testing.B) {
	debug = false
	scalar := getScalar()
	x := buildFP32(benchmarkRows, benchmarkCols)
	w := NewFloat32([]float32{scalar}, []int64{1})
	for i := 0; i < b.N; i++ {
		x.Div(w)
	}
}

func BenchmarkFP32DivScalarGo(b *testing.B) {
	debug = true
	scalar := getScalar()
	x := buildFP32(benchmarkRows, benchmarkCols)
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
	result := x.Div(NewFloat32(vec, []int64{cols})).Storage()
	if !equal(result, expect) {
		t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, result.Data())
	}
}

func TestFP32DivVectorGo(t *testing.T) {
	debug = true
	rows := getRows()
	cols := getCols()
	expect, vec := computeDivVector(rows, cols)
	x := buildFP32(rows, cols)
	result := x.Div(NewFloat32(vec, []int64{cols})).Storage()
	if !equal(result, expect) {
		t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, result.Data())
	}
}

func BenchmarkFP32DivVector(b *testing.B) {
	debug = false
	_, vec := computeDivVector(benchmarkRows, benchmarkCols)
	x := buildFP32(benchmarkRows, benchmarkCols)
	w := NewFloat32(vec, []int64{benchmarkCols})
	for i := 0; i < b.N; i++ {
		x.Div(w)
	}
}

func BenchmarkFP32DivVectorGo(b *testing.B) {
	debug = true
	_, vec := computeDivVector(benchmarkRows, benchmarkCols)
	x := buildFP32(benchmarkRows, benchmarkCols)
	w := NewFloat32(vec, []int64{benchmarkCols})
	for i := 0; i < b.N; i++ {
		x.Div(w)
	}
}
