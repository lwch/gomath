package half

import (
	"testing"
)

func TestEncode(t *testing.T) {
	if Encode(1) != 15360 {
		t.Error("Encode(1)!=15360")
	}
}

func TestDecode(t *testing.T) {
	if Decode(15360) != 1 {
		t.Error("Decode(15360)!=1")
	}
}
