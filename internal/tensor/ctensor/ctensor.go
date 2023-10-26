package ctensor

// #include "tensor.h"
import "C"

import (
	"github.com/lwch/gomath/internal/tensor"

	// for compile cpp files
	_ "github.com/lwch/gomath/internal/tensor/ctensor/fp16"
	_ "github.com/lwch/gomath/internal/tensor/ctensor/fp32"
)

type CTensor struct{}

var _ tensor.TensorImpl = &CTensor{}

func New() (*CTensor, bool) {
	ok := bool(C.tensor_init())
	if ok {
		return &CTensor{}, true
	}
	return nil, false
}
