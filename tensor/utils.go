package tensor

import (
	"github.com/lwch/gomath"
)

func computeInC() bool {
	switch {
	case gomath.HasAVX512():
		return true
	case gomath.HasAVX2():
		return true
	case gomath.HasAVX():
		return true
	// case gomath.HasSSE3():
	// 	return true
	// case gomath.HasSSSE3():
	// 	return true
	// case gomath.HasFMA3():
	// 	return true
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
