package tensor

import (
	"fmt"
	"runtime"

	"github.com/lwch/gomath"
)

var debug bool

type fnBuild func(shapes []int64) gomath.Tensor
type fnScalarAndVector func(scalar any, t gomath.Tensor, d int64) gomath.Tensor
type fnVectorAndVector func(ret, t, t2 gomath.Tensor, idx, row1, row2, d int64)

func matMul(t, t2 gomath.Tensor, build fnBuild, dotVector fnVectorAndVector) gomath.Tensor {
	size1, d1 := splitShapes(t)
	size2, d2 := splitShapes(t2)
	if d1 != d2 {
		panic(fmt.Errorf("dimension mismatch: %v and %v", t.Size(), t2.Size()))
	}
	if len(size1) == 0 && len(size2) == 0 {
		ret := build([]int64{1})
		dotVector(ret, t, t2, 0, 0, 0, d1)
		return ret
	}
	var m, n int64
	if len(size1) > 0 {
		m = size1[len(size1)-1]
		size1 = size1[:len(size1)-1]
	} else {
		m = 1
	}
	if len(size2) > 0 {
		n = size2[len(size2)-1]
		size2 = size2[:len(size2)-1]
	} else {
		n = 1
	}
	dims := maxSize(size1, size2)
	if dims == 0 {
		dims = 1
	}
	targetShapes := make([]int64, dims)
	for i := 0; i < dims; i++ {
		offset1 := len(size1) - i - 1
		offset2 := len(size2) - i - 1
		dim1 := int64(1)
		dim2 := int64(1)
		if offset1 >= 0 {
			dim1 = size1[offset1]
		}
		if offset2 >= 0 {
			dim2 = size2[offset2]
		}
		if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
			panic(fmt.Errorf("dimension mismatch: %v and %v", t.Size(), t2.Size()))
		}
		targetShapes[dims-i-1] = max(dim1, dim2)
	}
	targetGroups := sumShapes(targetShapes)
	group1 := sumShapes(size1) * m
	group2 := sumShapes(size2) * n
	ret := build(append(targetShapes, m, n))
	core := runtime.NumCPU()
	parallel(targetGroups, int64(core), func(offset, size int64, _ ...int64) {
		for group := offset; group < offset+size; group++ {
			idx := group * m * n
			parallel(m, int64(core), func(offset, size int64, args ...int64) {
				idx, group := args[0], args[1]
				for row := offset; row < offset+size; row++ {
					idx := idx + row*n
					parallel(n, int64(core), func(offset, size int64, args ...int64) {
						idx, group, row := args[0], args[1], args[2]
						for col := offset; col < offset+size; col++ {
							dotVector(ret, t, t2, idx+col, (group*m+row)%group1, (group*n+col)%group2, d1)
						}
					}, idx, group, row)
				}
			}, idx, group)
		}
	})
	return ret
}

func computeVectors(t, t2 gomath.Tensor,
	build fnBuild,
	scalar2Vector, vector2Scalar fnScalarAndVector,
	vector2Vector fnVectorAndVector) gomath.Tensor {
	size1, d1 := splitShapes(t)
	size2, d2 := splitShapes(t2)
	if len(size1) == 0 {
		return scalar2Vector(t.Storage().Get(0), t2, d2)
	}
	if len(size2) == 0 {
		return vector2Scalar(t2.Storage().Get(0), t, d1)
	}
	if d1 != d2 {
		panic(fmt.Errorf("dimension mismatch: %v and %v", t.Size(), t2.Size()))
	}
	dims := maxSize(size1, size2)
	if dims == 0 {
		dims = 1
	}
	targetShapes := make([]int64, dims)
	for i := 0; i < dims; i++ {
		offset1 := len(size1) - i - 1
		offset2 := len(size2) - i - 1
		dim1 := int64(1)
		dim2 := int64(1)
		if offset1 >= 0 {
			dim1 = size1[offset1]
		}
		if offset2 >= 0 {
			dim2 = size2[offset2]
		}
		if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
			panic(fmt.Errorf("dimension mismatch: %v and %v", t.Size(), t2.Size()))
		}
		targetShapes[dims-i-1] = max(dim1, dim2)
	}
	targetGroups := sumShapes(targetShapes)
	group1 := sumShapes(size1)
	group2 := sumShapes(size2)
	ret := build(append(targetShapes, d1))
	parallel(targetGroups, int64(runtime.NumCPU()), func(offset, size int64, _ ...int64) {
		for block := offset; block < offset+size; block++ {
			vector2Vector(ret, t, t2, block, block%group1, block%group2, d1)
		}
	})
	return ret
}
