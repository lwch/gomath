package tensor

import (
	"math"
	"math/rand"
	"testing"

	"github.com/lwch/gomath"
	"github.com/lwch/gomath/internal/half"
	"github.com/lwch/gomath/internal/tensor/ctensor"
	"github.com/lwch/gomath/internal/tensor/gotensor"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
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

func compareAndAssert(t *testing.T, result gomath.Storage, expect []float32, prefix string) {
	if !equal(result, expect) {
		switch data := result.Data().(type) {
		case []uint16:
			tmp := make([]float32, result.Size())
			half.DecodeArray(data, tmp)
			t.Fatalf("%s: expect=%v, got=%v", prefix, expect, tmp)
		case []float32:
			t.Fatalf("%s: expect=%v, got=%v", prefix, expect, data)
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

func testC(t *testing.T, fn func(*testing.T)) {
	useCTensor(func() {
		t.Run("C", func(t *testing.T) {
			fn(t)
		})
	})
}

func testGo(t *testing.T, fn func(*testing.T)) {
	useCTensor(func() {
		t.Run("Go", func(t *testing.T) {
			fn(t)
		})
	})
}

func benchmarkC(b *testing.B, name string, x, w gomath.Tensor, fn func(x, w gomath.Tensor)) {
	useCTensor(func() {
		b.Run(name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				fn(x, w)
			}
		})
	})
}

func benchmarkGo(b *testing.B, name string, x, w gomath.Tensor, fn func(x, w gomath.Tensor)) {
	useGoTensor(func() {
		b.Run(name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				fn(x, w)
			}
		})
	})
}

func benchmarkGoNum(b *testing.B, name string, fn func()) {
	b.Run(name, func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			fn()
		}
	})
}

func benchmarkMatMul(b *testing.B, goNum bool,
	buildVector func(int64) gomath.Tensor,
	buildMatrix func(int64, int64) gomath.Tensor,
	compute func(gomath.Tensor, gomath.Tensor)) {
	vector := buildFP16Vector(benchmarkCols)
	matrix := buildFP16Matrix(benchmarkRows, benchmarkCols)

	goNumVector := blas32.Vector{
		N:    benchmarkCols,
		Inc:  1,
		Data: make([]float32, benchmarkCols),
	}
	for i := range goNumVector.Data {
		goNumVector.Data[i] = float32(i) + 1
	}

	goNumMatrix := blas32.General{
		Rows:   benchmarkRows,
		Cols:   benchmarkCols,
		Stride: benchmarkCols,
		Data:   make([]float32, benchmarkRows*benchmarkCols),
	}
	for i := range goNumMatrix.Data {
		goNumMatrix.Data[i] = float32(i) + 1
	}

	benchmarkC(b, "Vector2Vector/C", vector, vector, compute)
	benchmarkGo(b, "Vector2Vector/Go", vector, vector, compute)
	if goNum {
		benchmarkGoNum(b, "Vector2Vector/GoNum", func() {
			blas32.Dot(goNumVector, goNumVector)
		})
	}

	benchmarkC(b, "Vector2Matrix/C", vector, matrix, compute)
	benchmarkGo(b, "Vector2Matrix/Go", vector, matrix, compute)
	if goNum {
		benchmarkGoNum(b, "Vector2Matrix/GoNum", func() {
			ret := blas32.Vector{
				N:    benchmarkCols,
				Inc:  1,
				Data: make([]float32, benchmarkCols),
			}
			blas32.Gemv(blas.Trans, 1, goNumMatrix, goNumVector, 0, ret)
		})
	}

	benchmarkC(b, "Matrix2Vector/C", matrix, vector, compute)
	benchmarkGo(b, "Matrix2Vector/Go", matrix, vector, compute)
	if goNum {
		benchmarkGoNum(b, "Matrix2Vector/GoNum", func() {
			ret := blas32.Vector{
				N:    benchmarkRows,
				Inc:  1,
				Data: make([]float32, benchmarkRows),
			}
			blas32.Gemv(blas.NoTrans, 1, goNumMatrix, goNumVector, 0, ret)
		})
	}

	benchmarkC(b, "Matrix2Matrix/C", matrix, matrix, compute)
	benchmarkGo(b, "Matrix2Matrix/Go", matrix, matrix, compute)
	if goNum {
		benchmarkGoNum(b, "Matrix2Matrix/GoNum", func() {
			ret := blas32.General{
				Rows:   benchmarkRows,
				Cols:   benchmarkRows,
				Stride: benchmarkRows,
				Data:   make([]float32, benchmarkRows*benchmarkRows),
			}
			blas32.Gemm(blas.NoTrans, blas.Trans, 1, goNumMatrix, goNumMatrix, 0, ret)
		})
	}
}

func batchBenchmark(b *testing.B,
	buildScalar func(float32) gomath.Tensor,
	buildVector func(int64) gomath.Tensor,
	buildMatrix func(int64, int64) gomath.Tensor,
	compute func(gomath.Tensor, gomath.Tensor)) {

	scalar := buildFP16Scalar(getScalar())
	vector := buildFP16Vector(benchmarkCols)
	matrix := buildFP16Matrix(benchmarkRows, benchmarkCols)

	benchmarkC(b, "Scalar2Scalar/C", scalar, scalar, compute)
	benchmarkGo(b, "Scalar2Scalar/Go", scalar, scalar, compute)

	benchmarkC(b, "Scalar2Vector/C", scalar, vector, compute)
	benchmarkGo(b, "Scalar2Vector/Go", scalar, vector, compute)

	benchmarkC(b, "Scalar2Matrix/C", scalar, matrix, compute)
	benchmarkGo(b, "Scalar2Matrix/Go", scalar, matrix, compute)

	benchmarkC(b, "Vector2Scalar/C", vector, scalar, compute)
	benchmarkGo(b, "Vector2Scalar/Go", vector, scalar, compute)

	benchmarkC(b, "Vector2Vector/C", vector, vector, compute)
	benchmarkGo(b, "Vector2Vector/Go", vector, vector, compute)

	benchmarkC(b, "Vector2Matrix/C", vector, matrix, compute)
	benchmarkGo(b, "Vector2Matrix/Go", vector, matrix, compute)

	benchmarkC(b, "Matrix2Scalar/C", matrix, scalar, compute)
	benchmarkGo(b, "Matrix2Scalar/Go", matrix, scalar, compute)

	benchmarkC(b, "Matrix2Vector/C", matrix, vector, compute)
	benchmarkGo(b, "Matrix2Vector/Go", matrix, vector, compute)

	benchmarkC(b, "Matrix2Matrix/C", matrix, matrix, compute)
	benchmarkGo(b, "Matrix2Matrix/Go", matrix, matrix, compute)
}
