package tensor

import (
	"testing"

	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
)

func testCompute2(t *testing.T,
	buildScalar func(float32) gomath.Tensor,
	buildVector func(int64) gomath.Tensor,
	buildMatrix func(int64, int64) gomath.Tensor,
	fn func(float32, float32) float32,
	op func(gomath.Tensor, gomath.Tensor) gomath.Tensor) {
	rows := getRows()
	cols := getCols()

	nScalar := getScalar()
	scalar := buildScalar(nScalar)

	vector := buildVector(cols)
	fp32Vector := vector.ToType(consts.Float32)
	vectorValues := fp32Vector.Storage().Data().([]float32)

	matrix := buildMatrix(rows, cols)
	fp32Matrix := matrix.ToType(consts.Float32)
	matrixValues := fp32Matrix.Storage().Data().([]float32)

	var tfn func(*testing.T)

	test := func(name string) {
		useCTensor(func() {
			t.Run(name+"/C", tfn)
		})
		useGoTensor(func() {
			t.Run(name+"/Go", tfn)
		})
	}

	tfn = func(t *testing.T) {
		value := fn(nScalar, nScalar)
		compareAndAssert(t, op(scalar, scalar).Storage(), []float32{value})
	}
	test("Scalar2Scalar")

	tfn = func(t *testing.T) {
		values := make([]float32, cols)
		for i := range values {
			values[i] = fn(nScalar, vectorValues[i])
		}
		compareAndAssert(t, op(scalar, vector).Storage(), values)
	}
	test("Scalar2Vector")

	tfn = func(t *testing.T) {
		values := make([]float32, cols)
		for i := range values {
			values[i] = fn(vectorValues[i], nScalar)
		}
		compareAndAssert(t, op(vector, scalar).Storage(), values)
	}
	test("Vector2Scalar")

	tfn = func(t *testing.T) {
		values := make([]float32, cols)
		for i := range values {
			values[i] = fn(vectorValues[i], vectorValues[i])
		}
		compareAndAssert(t, op(vector, vector).Storage(), values)
	}
	test("Vector2Vector")

	tfn = func(t *testing.T) {
		values := make([]float32, rows*cols)
		for i := range values {
			values[i] = fn(nScalar, matrixValues[i])
		}
		compareAndAssert(t, op(scalar, matrix).Storage(), values)
	}
	test("Scalar2Matrix")

	tfn = func(t *testing.T) {
		values := make([]float32, rows*cols)
		for i := range values {
			values[i] = fn(matrixValues[i], nScalar)
		}
		compareAndAssert(t, op(matrix, scalar).Storage(), values)
	}
	test("Matrix2Scalar")

	tfn = func(t *testing.T) {
		values := make([]float32, rows*cols)
		for i := range values {
			values[i] = fn(vectorValues[i%len(vectorValues)], matrixValues[i])
		}
		compareAndAssert(t, op(vector, matrix).Storage(), values)
	}
	test("Vector2Matrix")

	tfn = func(t *testing.T) {
		values := make([]float32, rows*cols)
		for i := range values {
			values[i] = fn(matrixValues[i], vectorValues[i%len(vectorValues)])
		}
		compareAndAssert(t, op(matrix, vector).Storage(), values)
	}
	test("Matrix2Vector")

	tfn = func(t *testing.T) {
		values := make([]float32, rows*cols)
		for i := range values {
			values[i] = fn(matrixValues[i], matrixValues[i])
		}
		compareAndAssert(t, op(matrix, matrix).Storage(), values)
	}
	test("Matrix2Matrix")
}

func testCompute11(t *testing.T, n float32,
	buildScalar func(float32) gomath.Tensor,
	buildVector func(int64) gomath.Tensor,
	buildMatrix func(int64, int64) gomath.Tensor,
	fn func(float32, float32) float32,
	op func(gomath.Tensor, float32) gomath.Tensor) {
	rows := getRows()
	cols := getCols()

	nScalar := getScalar()
	scalar := buildScalar(nScalar)

	vector := buildVector(cols)
	fp32Vector := vector.ToType(consts.Float32)
	vectorValues := fp32Vector.Storage().Data().([]float32)

	matrix := buildMatrix(rows, cols)
	fp32Matrix := matrix.ToType(consts.Float32)
	matrixValues := fp32Matrix.Storage().Data().([]float32)

	var tfn func(*testing.T)

	test := func(name string) {
		useCTensor(func() {
			t.Run(name+"/C", tfn)
		})
		useGoTensor(func() {
			t.Run(name+"/Go", tfn)
		})
	}

	tfn = func(t *testing.T) {
		value := fn(nScalar, n)
		compareAndAssert(t, op(scalar, n).Storage(), []float32{value})
	}
	test("Scalar")

	tfn = func(t *testing.T) {
		values := make([]float32, cols)
		for i := range values {
			values[i] = fn(vectorValues[i], n)
		}
		compareAndAssert(t, op(vector, n).Storage(), values)
	}
	test("Vector")

	tfn = func(t *testing.T) {
		values := make([]float32, rows*cols)
		for i := range values {
			values[i] = fn(matrixValues[i], n)
		}
		compareAndAssert(t, op(matrix, n).Storage(), values)
	}
	test("Matrix")
}
