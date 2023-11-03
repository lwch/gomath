package tensor

import (
	"fmt"
	"runtime"

	"github.com/lwch/gomath"
	"github.com/lwch/gomath/internal/tensor"
	"github.com/lwch/gomath/internal/tensor/ctensor"
	"github.com/lwch/gomath/internal/tensor/gotensor"
)

type rowdI interface {
	rowd(int64, int64) any
}

type tmpStorage interface {
	buildTmpStorage(int64) gomath.Storage
}

var impl tensor.TensorImpl
var goImpl tensor.TensorImpl

func init() {
	goImpl = gotensor.New()
	var ok bool
	impl, ok = ctensor.New()
	if !ok {
		impl = goImpl
	}
}

type fnBuild func(shapes []int64) gomath.Tensor
type fnScalarAndVector func(scalar any, t gomath.Tensor) gomath.Tensor
type fnVectorAndVector func(ret, x, w any)
type fnDotVector func(ret, x, w any, col int64)
type fnVectorCompute func(ret, x any)

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
			for r := offset; r < offset+size; r++ {
				idx := offset1 + r
				dTarget := row(ret, idx)
				dx := row(x, idx%group1)
				for col := int64(0); col < n; col++ {
					dw := row(w, (offset2+col)%group2)
					dotVector(dTarget, dx, dw, col)
				}
			}
		})
	}
	parallelW := func(offset1, offset2 int64) {
		parallel(n, int64(core), func(offset, size int64, args ...any) {
			for col := offset; col < offset+size; col++ {
				dw := row(w, (offset2+col)%group2)
				for r := int64(0); r < m; r++ {
					idx := offset1 + r
					dTarget := row(ret, idx)
					dx := row(x, idx%group1)
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

func compute11(x gomath.Tensor, build fnBuild, fn fnVectorCompute) gomath.Tensor {
	ret := build(x.Size())
	parallel(x.ElemSize(), int64(runtime.NumCPU()), func(offset, size int64, _ ...any) {
		dTarget := ret.Storage().Range(offset, offset+size)
		dx := x.Storage().Range(offset, offset+size)
		fn(dTarget, dx)
	})
	return ret
}

func compute2(x, w gomath.Tensor,
	build fnBuild,
	scalar2Vector, vector2Scalar fnScalarAndVector,
	vector2Vector fnVectorAndVector) gomath.Tensor {
	size1, d1 := splitShapes(x)
	size2, d2 := splitShapes(w)
	if len(size1) == 0 && d1 == 1 {
		return scalar2Vector(x.Storage().Get(0), w)
	}
	if len(size2) == 0 && d2 == 1 {
		return vector2Scalar(w.Storage().Get(0), x)
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
			dTarget := row(ret, block)
			dx := row(x, block%group1)
			dw := row(w, block%group2)
			vector2Vector(dTarget, dx, dw)
		}
	})
	return ret
}
