package gotensor

import "github.com/lwch/gomath/internal/tensor"

type GoTensor struct{}

var _ tensor.TensorImpl = &GoTensor{}

func New() *GoTensor {
	return &GoTensor{}
}
