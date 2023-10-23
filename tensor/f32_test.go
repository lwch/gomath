package tensor

import (
	"fmt"
	"testing"

	"github.com/lwch/gomath"
)

func matMul() gomath.Tensor {
	const rows = 64
	const cols = 4096
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = float32(i) + 1
	}
	x := NewFloat32(data, []int64{rows, cols})
	y := NewFloat32(data, []int64{rows, cols})
	return x.MatMul(y)
}

func TestF32MatMul(t *testing.T) {
	ts := matMul()
	fmt.Println(ts.Storage())
}

func BenchmarkMatMul(b *testing.B) {
	for i := 0; i < b.N; i++ {
		matMul()
	}
}

func TestF32MatMulGO(t *testing.T) {
	debug = true
	ts := matMul()
	fmt.Println(ts.Storage())
}

func BenchmarkMatMulGO(b *testing.B) {
	debug = true
	for i := 0; i < b.N; i++ {
		matMul()
	}
}
