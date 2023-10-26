package tensor

import (
	"fmt"
	"runtime"

	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
)

func (t *Float16) MatMul(t2 gomath.Tensor) gomath.Tensor {
	switch t2.Type() {
	case consts.Float16:
		return t.matMul(t2.(*Float16))
	case consts.Float32:
		return convert(t, consts.Float32).MatMul(t2)
	default:
		panic(ErrNotSupported)
	}
}

func (t *Float16) matMul(t2 *Float16) gomath.Tensor {
	size1, m, d1 := splitSize2(t)
	size2, n, d2 := splitSize2(t2)
	if d1 != d2 {
		panic(fmt.Errorf("dimension mismatch: %v and %v", t.Size(), t2.Size()))
	}
	if !sizeMatch(size1, size2) {
		panic(ErrBroadcast)
	}
	head := count(size1)
	if head == 0 {
		head = 1
	}
	data := make([]uint16, head*m*n)
	core := runtime.NumCPU()
	parallel(head, int64(core), func(_, offset, size int64) {
		for block := offset; block < offset+size; block++ {
			offset1 := block * m * d1
			offset2 := block * n * d1
			idx := block * m * n
			parallel(m, int64(core), func(_, offset, size int64) {
				for rows := offset; rows < offset+size; rows++ {
					offset1 := offset1 + rows*d1
					idx := idx + rows*n
					for cols := int64(0); cols < n; cols++ {
						offset2 := offset2 + cols*d1
						data[idx+cols] = t.impl.FP16DotVector(
							t.data[offset1:offset1+d1],
							t2.data[offset2:offset2+d1])
					}
				}
			})
		}
	})
	return NewFloat16Raw(data, append(size1, m, n),
		gomath.WithDevice(t.Device()))
}
