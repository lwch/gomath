package tensor

import (
	"testing"
)

func TestMatMulView(t *testing.T) {
	x := NewFloat32([]float32{1, 2, 3, 4, 5, 6}, []int64{2, 3})
	w := NewFloat32([]float32{1, 2, 3, 4, 5, 6}, []int64{6})
	x.MatMul(w.View(2, 3))
}

func TestViewMatMul(t *testing.T) {
	x := NewFloat32([]float32{1, 2, 3, 4, 5, 6}, []int64{6})
	w := NewFloat32([]float32{1, 2, 3, 4, 5, 6}, []int64{2, 3})
	x.View(2, 3).MatMul(w)
}

func TestAddView(t *testing.T) {
	x := NewFloat32([]float32{1, 2, 3, 4, 5, 6}, []int64{2, 3})
	w := NewFloat32([]float32{1, 2, 3, 4, 5, 6}, []int64{6})
	x.Add(w.View(2, 3))
}

func TestViewAdd(t *testing.T) {
	x := NewFloat32([]float32{1, 2, 3, 4, 5, 6}, []int64{6})
	w := NewFloat32([]float32{1, 2, 3, 4, 5, 6}, []int64{2, 3})
	x.View(2, 3).Add(w)
}

func TestSubView(t *testing.T) {
	x := NewFloat32([]float32{1, 2, 3, 4, 5, 6}, []int64{2, 3})
	w := NewFloat32([]float32{1, 2, 3, 4, 5, 6}, []int64{6})
	x.Sub(w.View(2, 3))
}

func TestViewSub(t *testing.T) {
	x := NewFloat32([]float32{1, 2, 3, 4, 5, 6}, []int64{6})
	w := NewFloat32([]float32{1, 2, 3, 4, 5, 6}, []int64{2, 3})
	x.View(2, 3).Sub(w)
}

func TestMulView(t *testing.T) {
	x := NewFloat32([]float32{1, 2, 3, 4, 5, 6}, []int64{2, 3})
	w := NewFloat32([]float32{1, 2, 3, 4, 5, 6}, []int64{6})
	x.Mul(w.View(2, 3))
}

func TestViewMul(t *testing.T) {
	x := NewFloat32([]float32{1, 2, 3, 4, 5, 6}, []int64{6})
	w := NewFloat32([]float32{1, 2, 3, 4, 5, 6}, []int64{2, 3})
	x.View(2, 3).Mul(w)
}

func TestDivView(t *testing.T) {
	x := NewFloat32([]float32{1, 2, 3, 4, 5, 6}, []int64{2, 3})
	w := NewFloat32([]float32{1, 2, 3, 4, 5, 6}, []int64{6})
	x.Div(w.View(2, 3))
}

func TestViewDiv(t *testing.T) {
	x := NewFloat32([]float32{1, 2, 3, 4, 5, 6}, []int64{6})
	w := NewFloat32([]float32{1, 2, 3, 4, 5, 6}, []int64{2, 3})
	x.View(2, 3).Div(w)
}
