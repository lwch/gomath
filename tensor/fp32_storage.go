package tensor

import (
	"github.com/lwch/gomath"
)

type Float32Storage []float32

var _ gomath.Storage = &Float32Storage{}

func NewFloat32Storage(data []float32) Float32Storage {
	return data
}

func (s Float32Storage) Data() any {
	return []float32(s)
}

func (s Float32Storage) Size() int {
	return len(s)
}

func (s Float32Storage) Get(i int64) any {
	return s[i]
}

func (s Float32Storage) Set(i int64, value any) {
	s[i] = value.(float32)
}

func (s Float32Storage) Range(fn func(int, any)) {
	for i, v := range s {
		fn(i, v)
	}
}
