package tensor

import (
	"math"
	"math/rand"
	"testing"

	"github.com/lwch/gomath"
	"github.com/lwch/gomath/internal/half"
	"github.com/lwch/gomath/internal/tensor/ctensor"
	"github.com/lwch/gomath/internal/tensor/gotensor"
)

const benchmarkRows = 64
const benchmarkCols = 4096

func getScalar() float32 {
	return float32(rand.Intn(10) + 1)
}

func getRows() int64 {
	return rand.Int63n(7) + 1
}

func getCols() int64 {
	var cols int64
	if gomath.HasAVX512() {
		cols = 16 + 8
	} else if gomath.HasAVX() {
		cols = 8 + 4
	} else {
		cols = 3
	}
	return cols
}

func equal(data gomath.Storage, expect []float32) bool {
	if data.Size() != len(expect) {
		return false
	}
	ok := true
	var std float32
	data.Range(func(i int, a any) {
		v, ok := a.(float32)
		if !ok {
			v = half.Decode(a.(uint16))
		}
		std += float32(math.Pow(float64(v-expect[i]), 2))
		if v != expect[i] {
			ok = false
		}
	})
	if ok {
		return true
	}
	std /= float32(data.Size())
	return std < 0.01
}

func compareAndAssert(t *testing.T, result gomath.Storage, expect []float32) {
	if !equal(result, expect) {
		switch data := result.Data().(type) {
		case []uint16:
			tmp := make([]float32, result.Size())
			half.DecodeArray(data, tmp)
			t.Fatalf("expect=%v, got=%v", expect, tmp)
		case []float32:
			t.Fatalf("expect=%v, got=%v", expect, data)
		}
	}
}

func useCTensor(fn func()) {
	old := impl
	var ok bool
	impl, ok = ctensor.New()
	if !ok {
		panic("can not initialize ctensor")
	}
	fn()
	impl = old
}

func useGoTensor(fn func()) {
	old := impl
	impl = gotensor.New()
	fn()
	impl = old
}
