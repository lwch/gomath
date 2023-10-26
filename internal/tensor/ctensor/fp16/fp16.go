package fp16

// #cgo CXXFLAGS: -mavx -mavx2 -mavx512f -mavx512dq -march=native -O3 -std=c++11 -Wno-ignored-attributes
import "C"
