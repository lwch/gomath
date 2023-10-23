package gomath

import (
	"fmt"
	"testing"
)

func TestGetInfo(t *testing.T) {
	fmt.Println("HasAVX:", HasAVX())
	fmt.Println("HasAVX2:", HasAVX2())
	fmt.Println("HasAVX512:", HasAVX512())
	fmt.Println("HasSSE3:", HasSSE3())
	fmt.Println("HasSSSE3:", HasSSSE3())
	fmt.Println("HasFMA3:", HasFMA3())
}
