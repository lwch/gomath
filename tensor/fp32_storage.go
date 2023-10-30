package tensor

import (
	"github.com/lwch/gomath"
)

type Float32Storage struct {
	d    int64
	data []float32
}

var _ gomath.Storage = &Float32Storage{}

func NewFloat32Storage(data []float32, d int64) *Float32Storage {
	return &Float32Storage{
		d:    d,
		data: data,
	}
}

func (s Float32Storage) Data() any {
	return s.data
}

func (s Float32Storage) Size() int {
	return len(s.data)
}

func (s Float32Storage) Get(i int64) any {
	return s.data[i]
}

func (s Float32Storage) Set(i int64, value any) {
	s.data[i] = value.(float32)
}

func (s Float32Storage) Range(fn func(int, any)) {
	for i, v := range s.data {
		fn(i, v)
	}
}

func (s Float32Storage) Row(i int64) any {
	return s.data[i*s.d : (i+1)*s.d]
}

func (s Float32Storage) Copy(data any) {
	if src, ok := data.([]float32); ok {
		copy(s.data, src)
	}
}
