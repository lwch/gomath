package gotensor

func dotVector[T float32](x, w []T, d int64) T {
	var ret T
	for i := int64(0); i < d; i++ {
		ret += x[i] * w[i]
	}
	return ret
}
