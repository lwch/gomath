package tensor

import (
	"github.com/lwch/gomath"
)

type Float16Storage []uint16

var _ gomath.Storage = &Float16Storage{}

func NewFloat16Storage(data []uint16) Float16Storage {
	return Float16Storage(data)
}

func (s Float16Storage) Data() any {
	return []uint16(s)
}

func (s Float16Storage) Size() int {
	return len(s)
}

func (s Float16Storage) Get(i int64) any {
	return s[i]
}

func (s Float16Storage) Set(i int64, value any) {
	s[i] = value.(uint16)
}

func (s Float16Storage) Range(fn func(int, any)) {
	for i, v := range s {
		fn(i, v)
	}
}
