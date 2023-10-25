package tensor

import (
	"sync"

	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
)

func computeInC(t consts.Type) bool {
	if debug {
		return false
	}
	switch t {
	case consts.Float16:
		return fp16ComputeInC
	case consts.Float32:
		return fp32ComputeInC
	default:
		return false
	}
}

func splitSize(t gomath.Tensor) ([]int64, int64, int64) {
	size := t.Size()
	return size[:len(size)-2], size[len(size)-2], size[len(size)-1]
}

func count(size []int64) int64 {
	if len(size) == 0 {
		return 0
	}
	ret := size[0]
	for i := 1; i < len(size); i++ {
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
	wg.Add(int(batches))
	for i := int64(0); i < batches; i++ {
		offset := i * step
		if i == batches-1 {
			step = size - offset
		}
		go func(i, offset, step int64) {
			defer wg.Done()
			fn(i, offset, step)
		}(i, offset, step)
	}
	wg.Wait()
}

func dotVector[T float32](x, w []T, d int64) T {
	var ret T
	for i := int64(0); i < d; i++ {
		ret += x[i] * w[i]
	}
	return ret
}
