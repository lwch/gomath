package tensor

import (
	"testing"

	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
)

func dotVector(x, w []float32) float32 {
	var sum float32
	for i := range x {
		sum += x[i] * w[i]
	}
	return sum
}

func testMatMul(t *testing.T,
	buildVector func(int64) gomath.Tensor,
	buildMatrix func(int64, int64) gomath.Tensor) {
	rows := getRows()
	cols := getCols()

	vector := buildVector(cols)
	fp32Vector := vector.ToType(consts.Float32)
	vectorValues := fp32Vector.Storage().Data().([]float32)

	matrix := buildMatrix(rows, cols)
	fp32Matrix := matrix.ToType(consts.Float32)

	var tfn func(*testing.T)

	test := func(name string) {
		useCTensor(func() {
			t.Run(name+"/C", tfn)
		})
		useGoTensor(func() {
			t.Run(name+"/Go", tfn)
		})
	}

	// vector @ vector
	tfn = func(t *testing.T) {
		value := dotVector(vectorValues, vectorValues)
		compareAndAssert(t, vector.MatMul(vector).Storage(), []float32{value})
	}
	test("Vector2Vector")

	// vector @ matrix(broadcast)
	tfn = func(t *testing.T) {
		values := make([]float32, rows)
		for i := int64(0); i < rows; i++ {
			values[i] = dotVector(vectorValues, row(fp32Matrix, i).([]float32))
		}
		compareAndAssert(t, vector.MatMul(matrix).Storage(), values)
	}
	test("Vector2Matrix")

	// matrix @ vector(broadcast)
	tfn = func(t *testing.T) {
		values := make([]float32, rows)
		for i := range values {
			values[i] = dotVector(row(fp32Matrix, int64(i)).([]float32), vectorValues)
		}
		compareAndAssert(t, matrix.MatMul(vector).Storage(), values)
	}
	test("Matrix2Vector")

	// matrix @ matrix
	tfn = func(t *testing.T) {
		values := make([]float32, rows*rows)
		for r := int64(0); r < rows; r++ {
			for c := int64(0); c < rows; c++ {
				values[r*rows+c] = dotVector(
					row(fp32Matrix, r).([]float32),
					row(fp32Matrix, c).([]float32))
			}
		}
		compareAndAssert(t, matrix.MatMul(matrix).Storage(), values)
	}
	test("Matrix2Matrix")
}
