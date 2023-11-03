package fp32

// #cgo CXXFLAGS: -mavx -mavx2 -mavx512f -mavx512dq -march=native -O3 -Wno-ignored-attributes
// #cgo LDFLAGS: -lm
import "C"
