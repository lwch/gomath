package tensor

import (
	"sync"

	"github.com/lwch/gomath"
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
