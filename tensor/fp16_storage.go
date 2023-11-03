package tensor

import (
	"github.com/lwch/gomath"
	"github.com/lwch/gomath/consts"
)

type Float16Storage []uint16

var _ gomath.Storage = &Float16Storage{}
var _ rowdI = &Float16Storage{}
var _ tmpStorage = &Float16Storage{}

func NewFloat16Storage(data []uint16) Float16Storage {
	return Float16Storage(data)
}

func (s Float16Storage) Type() consts.Type {
	return consts.Float16
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

func (s Float16Storage) Loop(fn func(int, any)) {
	for i, v := range s {
		fn(i, v)
	}
}

func (s Float16Storage) Range(i, j int64) any {
	return []uint16(s[i:j])
}

func (s Float16Storage) rowd(i, d int64) any {
	return []uint16(s[i*d : (i+1)*d])
}

func (s Float16Storage) buildTmpStorage(size int64) gomath.Storage {
	return Float16Storage(make([]uint16, size))
}
