package tensor

import (
	"fmt"
	"runtime"

	"github.com/lwch/gomath"
	"github.com/lwch/gomath/internal/tensor"
	"github.com/lwch/gomath/internal/tensor/ctensor"
	"github.com/lwch/gomath/internal/tensor/gotensor"
)

var impl tensor.TensorImpl
var goImpl tensor.TensorImpl

func init() {
	var ok bool
	impl, ok = ctensor.New()
	if !ok {
		impl = gotensor.New()
	}
	goImpl = gotensor.New()
}

type fnBuild func(shapes []int64) gomath.Tensor
type fnScalarAndVector func(impl tensor.TensorImpl, scalar any, t gomath.Tensor, d int64) gomath.Tensor
type fnVectorAndVector func(ret, x, w any)
type fnDotVector func(ret, x, w any, col int64)

func matMul(x, w gomath.Tensor, build fnBuild, dotVector fnDotVector) gomath.Tensor {
	size1, d1 := splitShapes(x)
	size2, d2 := splitShapes(w)
	if d1 != d2 {
		panic(fmt.Errorf("dimension mismatch: %v and %v", x.Size(), w.Size()))
	}
	if len(size1) == 0 && len(size2) == 0 {
		ret := build([]int64{1})
		dotVector(
			ret.Storage().Data(),
			x.Storage().Data(),
			w.Storage().Data(), 0)
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
			panic(fmt.Errorf("dimension mismatch: %v and %v", x.Size(), w.Size()))
		}
		targetShapes[dims-i-1] = max(dim1, dim2)
	}
	targetGroups := sumShapes(targetShapes)
	group1 := sumShapes(size1) * m
	group2 := sumShapes(size2) * n
	ret := build(append(targetShapes, m, n))
	core := runtime.NumCPU()
	parallelX := func(offset1, offset2 int64) {
		parallel(m, int64(core), func(offset, size int64, args ...any) {
			for row := offset; row < offset+size; row++ {
				idx := offset1 + row
				dTarget := ret.Storage().Row(idx)
				dx := x.Storage().Row(idx % group1)
				for col := int64(0); col < n; col++ {
					dw := w.Storage().Row((offset2 + col) % group2)
					dotVector(dTarget, dx, dw, col)
				}
			}
		})
	}
	parallelW := func(offset1, offset2 int64) {
		parallel(n, int64(core), func(offset, size int64, args ...any) {
			for col := offset; col < offset+size; col++ {
				dw := w.Storage().Row((offset2 + col) % group2)
				for row := int64(0); row < m; row++ {
					idx := offset1 + row
					dTarget := ret.Storage().Row(idx)
					dx := x.Storage().Row(idx % group1)
					dotVector(dTarget, dx, dw, col)
				}
			}
		})
	}
	var pv func(offset1, offset2 int64)
	if m > n {
		pv = parallelX
	} else {
		pv = parallelW
	}
	parallel(targetGroups, int64(core), func(offset, size int64, _ ...any) {
		offset1 := offset * m
		offset2 := offset * n
		for group := offset; group < offset+size; group++ {
			pv(offset1, offset2)
			offset1 += m
			offset2 += n
		}
	})
	return ret
}

func computeVectors(x, w gomath.Tensor,
	build fnBuild,
	scalar2Vector, vector2Scalar fnScalarAndVector,
	vector2Vector fnVectorAndVector) gomath.Tensor {
	size1, d1 := splitShapes(x)
	size2, d2 := splitShapes(w)
	if len(size1) == 0 && d1 == 1 {
		if len(size2) == 0 || len(size2) == 1 {
			return scalar2Vector(goImpl, x.Storage().Get(0), w, d2)
		}
		return scalar2Vector(impl, x.Storage().Get(0), w, d2)
	}
	if len(size2) == 0 && d2 == 1 {
		if len(size1) == 0 || len(size1) == 1 {
			return vector2Scalar(goImpl, w.Storage().Get(0), x, d1)
		}
		return vector2Scalar(impl, w.Storage().Get(0), x, d1)
	}
	if d1 != d2 {
		panic(fmt.Errorf("dimension mismatch: %v and %v", x.Size(), w.Size()))
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
			panic(fmt.Errorf("dimension mismatch: %v and %v", x.Size(), w.Size()))
		}
		targetShapes[dims-i-1] = max(dim1, dim2)
	}
	targetGroups := sumShapes(targetShapes)
	group1 := sumShapes(size1)
	group2 := sumShapes(size2)
	ret := build(append(targetShapes, d1))
	parallel(targetGroups, int64(runtime.NumCPU()), func(offset, size int64, _ ...any) {
		for block := offset; block < offset+size; block++ {
			dTarget := ret.Storage().Row(block)
			dx := x.Storage().Row(block % group1)
			dw := w.Storage().Row(block % group2)
			vector2Vector(dTarget, dx, dw)
		}
	})
	return ret
}
