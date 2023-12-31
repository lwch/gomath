package consts

type Type byte

const (
	Float16 Type = iota
	Float32
)

type Device byte

const (
	CPU Device = iota
	MPS
)
