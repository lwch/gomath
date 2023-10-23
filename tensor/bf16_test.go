package tensor

import "testing"

func TestBF16MatMul(t *testing.T) {
	x := NewBFloat16([]uint16{1, 2, 3, 4, 5, 6}, []int64{2, 3})
	y := NewBFloat16([]uint16{1, 2, 3, 4, 5, 6}, []int64{2, 3})
	x.MatMul(y)
}
