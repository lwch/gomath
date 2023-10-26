package tensor

import (
	"fmt"
	"runtime"

	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
)

// MatMul (..., m, d) @ (..., n, d) -> (..., m, n)
func (t *Float32) MatMul(t2 gomath.Tensor) gomath.Tensor {
	switch t2.Type() {
	case consts.Float16:
		return t.matMul(convert(t2, consts.Float32).(*Float32))
	case consts.Float32:
		return t.matMul(t2.(*Float32))
	default:
		panic("not supported")
	}
}

func (t *Float32) matMul(t2 *Float32) gomath.Tensor {
	size1, m, d1 := splitSize2(t)
	size2, n, d2 := splitSize2(t2)
	if d1 != d2 {
		panic(fmt.Errorf("dimension mismatch: %v and %v", t.Size(), t2.Size()))
	}
	if !sizeMatch(size1, size2) {
		panic(ErrBroadcast)
	}
	head := count(size1)
	data := make([]float32, head*m*n)
	core := runtime.NumCPU()
	parallel(head, int64(core), func(_, offset, size int64) {
		for block := offset; block < offset+size; block++ {
			offset1 := block * m * d1
			offset2 := block * n * d2
			idx := block * m * n
			parallel(m, int64(core), func(batch, offset, size int64) {
				for rows := offset; rows < offset+size; rows++ {
					offset1 := offset1 + rows*d1
					idx := idx + rows*n
					for cols := int64(0); cols < n; cols++ {
						offset2 := offset2 + cols*d2
						data[idx+cols] = t.impl.FP32DotVector(
							t.data[offset1:offset1+d1],
							t2.data[offset2:offset2+d2])
					}
				}
			})
		}
	})
	return NewFloat32(data, append(size1, m, n),
		gomath.WithDevice(t.Device()))
}
