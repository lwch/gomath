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
	useCTensor(func() {
		rows := getRows()
		cols := getCols()
		expect := computeMatMul(rows, cols)
		x := buildFP16(rows, cols)
		result := x.MatMul(x).Storage()
		if !equal(result, expect) {
			tmp := make([]float32, result.Size())
			half.DecodeArray(result.Data().([]uint16), tmp)
			t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, tmp)
		}
	})
}

func BenchmarkFP16MatMul(b *testing.B) {
	useCTensor(func() {
		x := buildFP16(benchmarkRows, benchmarkCols)
		for i := 0; i < b.N; i++ {
			x.MatMul(x)
		}
	})
}

func TestFP16MatMulGo(t *testing.T) {
	useGoTensor(func() {
		rows := getRows()
		cols := getCols()
		expect := computeMatMul(rows, cols)
		x := buildFP16(rows, cols)
		result := x.MatMul(x).Storage()
		if !equal(result, expect) {
			tmp := make([]float32, result.Size())
			half.DecodeArray(result.Data().([]uint16), tmp)
			t.Fatalf("(%d, %d): expect=%v, got=%v", rows, cols, expect, tmp)
		}
	})
}

func BenchmarkFP16MatMulGo(b *testing.B) {
	useGoTensor(func() {
		x := buildFP16(benchmarkRows, benchmarkCols)
		for i := 0; i < b.N; i++ {
			x.MatMul(x)
		}
	})
}
