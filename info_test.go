package gomath

import (
	"fmt"
	"testing"
)

func TestGetInfo(t *testing.T) {
	fmt.Println("HasAVX:", HasAVX())
	fmt.Println("HasAVX512:", HasAVX512())
}
