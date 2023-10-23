package gomath

// #include <stdbool.h>
// #include "info.h"
import "C"

func HasAVX() bool {
	return bool(C.has_avx())
}

func HasAVX512() bool {
	return bool(C.has_avx512())
}
