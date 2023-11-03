package tensor

import (
	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
)

type Float32Storage []float32

var _ gomath.Storage = &Float32Storage{}
var _ rowdI = &Float32Storage{}
var _ tmpStorage = &Float32Storage{}

func NewFloat32Storage(data []float32) Float32Storage {
	return data
}

func (s Float32Storage) Type() consts.Type {
	return consts.Float32
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

func (s Float32Storage) Loop(fn func(int, any)) {
	for i, v := range s {
		fn(i, v)
	}
}

func (s Float32Storage) Range(i, j int64) any {
	return []float32(s[i:j])
}

func (s Float32Storage) rowd(i, d int64) any {
	return []float32(s[i*d : (i+1)*d])
}

func (s Float32Storage) buildTmpStorage(size int64) gomath.Storage {
	return Float32Storage(make([]float32, size))
}
