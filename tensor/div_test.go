package tensor

import (
	"testing"

	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
)

func scalarDivVector(s float32, values []float32) []float32 {
	ret := make([]float32, len(values))
	for i := range values {
		ret[i] = s / values[i]
	}
	return ret
}

func vectorDivScalar(values []float32, s float32) []float32 {
	ret := make([]float32, len(values))
	for i := range values {
		ret[i] = values[i] / s
	}
	return ret
}

func divVector(x, w []float32) []float32 {
	ret := make([]float32, len(x))
	for i := range x {
		ret[i] = x[i] / w[i]
	}
	return ret
}

func testDiv(t *testing.T,
	buildScalar func(float32) gomath.Tensor,
	buildVector func(int64) gomath.Tensor,
	buildMatrix func(int64, int64) gomath.Tensor) {
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

	// scalar / scalar
	value := nScalar / nScalar
	compareAndAssert(t, scalar.Mul(scalar).Storage(), []float32{value}, "scalar / scalar")

	// scalar / vector(broadcast)
	values := scalarDivVector(nScalar, vectorValues)
	compareAndAssert(t, scalar.Mul(vector).Storage(), values, "scalar / vector")

	// vector / scalar(broadcast)
	values = vectorDivScalar(vectorValues, nScalar)
	compareAndAssert(t, vector.Mul(scalar).Storage(), values, "vector / scalar")

	// vector / vector
	values = divVector(vectorValues, vectorValues)
	compareAndAssert(t, vector.Mul(vector).Storage(), values, "vector / vector")

	// scalar / matrix(broadcast)
	values = scalarDivVector(nScalar, matrixValues)
	compareAndAssert(t, scalar.Mul(matrix).Storage(), values, "scalar / matrix")

	// matrix / scalar(broadcast)
	values = vectorDivScalar(matrixValues, nScalar)
	compareAndAssert(t, matrix.Mul(scalar).Storage(), values, "matrix / scalar")

	// vector / matrix(broadcast)
	values = make([]float32, rows*cols)
	for i := int64(0); i < rows; i++ {
		tmp := divVector(vectorValues, fp32Matrix.Storage().Row(i).([]float32))
		copy(values[i*cols:(i+1)*cols], tmp)
	}
	compareAndAssert(t, vector.Mul(matrix).Storage(), values, "vector / matrix")

	// matrix / vector(broadcast)
	values = make([]float32, rows*cols)
	for i := int64(0); i < rows; i++ {
		tmp := divVector(fp32Matrix.Storage().Row(i).([]float32), vectorValues)
		copy(values[i*cols:(i+1)*cols], tmp)
	}
	compareAndAssert(t, matrix.Mul(vector).Storage(), values, "matrix / vector")

	// matrix / matrix
	values = divVector(matrixValues, matrixValues)
	compareAndAssert(t, matrix.Mul(matrix).Storage(), values, "matrix * matrix")
}
