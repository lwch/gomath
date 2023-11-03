package tensor

import (
	"testing"

	"github.com/lwch/gomath"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
)

func benchmarkC11(b *testing.B, name string, x gomath.Tensor, n float32, fn func(x gomath.Tensor, n float32)) {
	useCTensor(func() {
		b.Run(name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				fn(x, n)
			}
		})
	})
}

func benchmarkGo11(b *testing.B, name string, x gomath.Tensor, n float32, fn func(x gomath.Tensor, n float32)) {
	useGoTensor(func() {
		b.Run(name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				fn(x, n)
			}
		})
	})
}

func benchmarkC2(b *testing.B, name string, x, w gomath.Tensor, fn func(x, w gomath.Tensor)) {
	useCTensor(func() {
		b.Run(name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				fn(x, w)
			}
		})
	})
}

func benchmarkGo2(b *testing.B, name string, x, w gomath.Tensor, fn func(x, w gomath.Tensor)) {
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

	benchmark := func(name string, x, w gomath.Tensor) {
		benchmarkC2(b, name+"/C", x, w, compute)
		benchmarkGo2(b, name+"/Go", x, w, compute)
	}

	benchmark("Vector2Vector", vector, vector)
	if goNum {
		benchmarkGoNum(b, "Vector2Vector/GoNum", func() {
			blas32.Dot(goNumVector, goNumVector)
		})
	}

	benchmark("Vector2Matrix", vector, matrix)
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

	benchmark("Matrix2Vector", matrix, vector)
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

	benchmark("Matrix2Matrix", matrix, matrix)
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

func benchmarkCompute2(b *testing.B,
	buildScalar func(float32) gomath.Tensor,
	buildVector func(int64) gomath.Tensor,
	buildMatrix func(int64, int64) gomath.Tensor,
	compute func(gomath.Tensor, gomath.Tensor)) {

	scalar := buildFP16Scalar(getScalar())
	vector := buildFP16Vector(benchmarkCols)
	matrix := buildFP16Matrix(benchmarkRows, benchmarkCols)

	benchmark := func(name string, x, w gomath.Tensor) {
		benchmarkC2(b, name+"/C", x, w, compute)
		benchmarkGo2(b, name+"/Go", x, w, compute)
	}

	benchmark("Scalar2Scalar", scalar, scalar)
	benchmark("Scalar2Vector", scalar, vector)
	benchmark("Scalar2Matrix", scalar, matrix)
	benchmark("Vector2Scalar", vector, scalar)
	benchmark("Vector2Vector", vector, vector)
	benchmark("Vector2Matrix", vector, matrix)
	benchmark("Matrix2Scalar", matrix, scalar)
	benchmark("Matrix2Vector", matrix, vector)
	benchmark("Matrix2Matrix", matrix, matrix)
}

func benchmarkCompute11(b *testing.B, n float32,
	buildScalar func(float32) gomath.Tensor,
	buildVector func(int64) gomath.Tensor,
	buildMatrix func(int64, int64) gomath.Tensor,
	compute func(gomath.Tensor, float32)) {

	scalar := buildFP16Scalar(getScalar())
	vector := buildFP16Vector(benchmarkCols)
	matrix := buildFP16Matrix(benchmarkRows, benchmarkCols)

	benchmark := func(name string, x gomath.Tensor) {
		benchmarkC11(b, name+"/C", x, n, compute)
		benchmarkGo11(b, name+"/Go", x, n, compute)
	}

	benchmark("Scalar", scalar)
	benchmark("Vector", vector)
	benchmark("Matrix", matrix)
}
