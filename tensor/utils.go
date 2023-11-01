package tensor

import (
	"sync"

	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
	"github.com/lwch/gomath/internal/utils"
)

func splitShapes(t gomath.Tensor) ([]int64, int64) {
	size := t.Size()
	return size[:len(size)-1], size[len(size)-1]
}

func sumShapes(size []int64) int64 {
	ret := int64(1)
	for i := 0; i < len(size); i++ {
		ret *= size[i]
	}
	return ret
}

func parallel(size, batches int64, fn func(offset, size int64, args ...any), args ...any) {
	var step int64
	if size%batches == 0 {
		step = size / batches
	} else {
		step = int64(float64(size)/float64(batches) + 0.5)
	}
	var wg sync.WaitGroup
	for i := int64(0); i < batches; i++ {
		offset := i * step
		if i == batches-1 {
			step = size - offset
		}
		if offset+step >= size {
			step = size - offset
		}
		if step == 0 {
			continue
		}
		wg.Add(1)
		go func(offset, step int64) {
			defer wg.Done()
			fn(offset, step, args...)
		}(offset, step)
	}
	wg.Wait()
}

func max(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

func maxSize(size1, size2 []int64) int {
	if len(size1) > len(size2) {
		return len(size1)
	}
	return len(size2)
}

func isContiguous(shape, stride []int64) bool {
	if stride[len(stride)-1] != 1 {
		return false
	}
	for i := 0; i < len(shape)-1; i++ {
		if stride[i] != shape[i+1]*stride[i+1] {
			return false
		}
	}
	return true
}

func contiguous(t gomath.Tensor) gomath.Tensor {
	if isContiguous(t.Size(), t.Stride()) {
		return t
	}
	cnt := t.ElemSize()
	s := t.Storage().(tmpStorage).buildTmpStorage(cnt)
	for i := int64(0); i < cnt; i++ {
		idx := make([]int64, t.Dim())
		for j := t.Dim() - 1; j >= 0; j-- {
			idx[j] = i % t.Size()[j]
			i = (i - t.Size()[j]) / t.Size()[j]
		}
		var offset int64
		for j := 0; j < len(idx); j++ {
			offset += idx[j] * t.Stride()[j]
		}
		s.Set(i, t.Storage().Get(offset))
	}
	switch t.Type() {
	case consts.Float16:
		return NewFloat16WithStorage(s.(Float16Storage), t.Size(), gomath.WithDevice(t.Device()))
	case consts.Float32:
		return NewFloat32WithStorage(s.(Float32Storage), t.Size(), gomath.WithDevice(t.Device()))
	default:
		panic(ErrNotSupported)
	}
}

func row(t gomath.Tensor, i int64) any {
	d := t.Size()[t.Dim()-1]
	if isContiguous(t.Size(), t.Stride()) {
		return t.Storage().(rowdI).rowd(i, d)
	}
	idx := make([]int64, t.Dim()-1)
	for j := t.Dim() - 1; j > 0; j-- {
		idx[j] = i % t.Size()[j]
		i = (i - t.Size()[j]) / t.Size()[j]
	}
	var offset int64
	for j := 0; j < len(idx); j++ {
		offset += idx[j] * t.Stride()[j]
	}
	tmp := t.Storage().(tmpStorage).buildTmpStorage(d)
	for j := int64(0); j < d; j++ {
		tmp.Set(j, t.Storage().Get(offset+j*t.Stride()[t.Dim()-1]))
	}
	return tmp.Data()
}

func get(idx, stride []int64, store gomath.Storage) any {
	if len(idx) != len(stride) {
		panic("invalid index")
	}
	var offset int64
	for i := 0; i < len(idx); i++ {
		offset += idx[i] * stride[i]
	}
	return store.Get(offset)
}

func view(t gomath.Tensor, shape []int64) gomath.Tensor {
	if t.ElemSize() != sumShapes(shape) {
		panic("invalid shape")
	}
	stride := utils.ComputeStride(shape)
	return &View{
		device: t.Device(),
		shape:  shape,
		stride: stride,
		store:  t.Storage(),
	}
}

func transpose(t gomath.Tensor, dim0, dim1 int64) gomath.Tensor {
	if dim0 >= t.Dim() || dim1 >= t.Dim() {
		panic("invalid dim")
	}
	if dim0 == dim1 {
		return t
	}
	shape := t.Size()
	stride := t.Stride()
	shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
	stride[dim0], stride[dim1] = stride[dim1], stride[dim0]
	return &View{
		device: t.Device(),
		shape:  shape,
		stride: stride,
		store:  t.Storage(),
	}
}

func reshape(t gomath.Tensor, shape []int64) gomath.Tensor {
	if t.ElemSize() != sumShapes(shape) {
		panic("invalid shape")
	}
	return view(t, shape).Contiguous()
}
