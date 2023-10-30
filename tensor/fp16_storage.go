package tensor

import (
	"github.com/lwch/gomath"
	"github.com/lwch/gomath/internal/half"
)

type Float16Storage struct {
	d    int64
	data []uint16
}

var _ gomath.Storage = &Float16Storage{}

func NewFloat16Storage(data []uint16, d int64) *Float16Storage {
	return &Float16Storage{
		d:    d,
		data: data,
	}
}

func (s Float16Storage) Data() any {
	return s.data
}

func (s Float16Storage) Size() int {
	return len(s.data)
}

func (s Float16Storage) Get(i int64) any {
	return s.data[i]
}

func (s Float16Storage) Set(i int64, value any) {
	s.data[i] = value.(uint16)
}

func (s Float16Storage) Range(fn func(int, any)) {
	for i, v := range s.data {
		fn(i, v)
	}
}

func (s Float16Storage) Row(i int64) any {
	return s.data[i*s.d : (i+1)*s.d]
}

func (s Float16Storage) Copy(data any) {
	if src, ok := data.([]uint16); ok {
		copy(s.data, src)
	} else if src, ok := data.([]float32); ok {
		for i, v := range src {
			s.data[i] = half.Encode(v)
		}
	}
}
