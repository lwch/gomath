package consts

type Type byte

const (
	BFloat16 Type = iota
	Float16
	Float32
	Float64
)

type Device byte

const (
	CPU Device = iota
	GPU
)
