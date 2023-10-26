package tensor

import (
	"sync"

	"github.com/lwch/gomath"
)

func splitSize1(t gomath.Tensor) ([]int64, int64) {
	size := t.Size()
	return size[:len(size)-1], size[len(size)-1]
}

func splitSize2(t gomath.Tensor) ([]int64, int64, int64) {
	size := t.Size()
	return size[:len(size)-2], size[len(size)-2], size[len(size)-1]
}

func count(size []int64) int64 {
	ret := int64(1)
	for i := 0; i < len(size); i++ {
		ret *= size[i]
	}
	return ret
}

func sizeMatch(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i, v := range a {
		if v != b[i] {
			return false
		}
	}
	return true
}

func parallel(size, batches int64, fn func(batch, offset, size int64)) {
	step := size / batches
	var wg sync.WaitGroup
	for i := int64(0); i < batches; i++ {
		offset := i * step
		if i == batches-1 {
			step = size - offset
		}
		if step == 0 {
			continue
		}
		wg.Add(1)
		go func(i, offset, step int64) {
			defer wg.Done()
			fn(i, offset, step)
		}(i, offset, step)
	}
	wg.Wait()
}
