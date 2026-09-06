package harness

// DetectImageMimeType sniffs a buffer and returns the MIME type for supported
// image formats (jpeg, png, gif, webp, bmp), or "" when unsupported.
func DetectImageMimeType(b []byte) string {
	if len(b) < 8 {
		return ""
	}
	switch {
	case startsWith(b, []byte{0xff, 0xd8, 0xff}):
		if len(b) > 3 && b[3] == 0xf7 {
			return ""
		}
		return "image/jpeg"
	case startsWith(b, []byte{0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a}):
		if isPng(b) && !isAnimatedPng(b) {
			return "image/png"
		}
		return ""
	case startsWithAscii(b, 0, "GIF"):
		return "image/gif"
	case startsWithAscii(b, 0, "RIFF") && startsWithAscii(b, 8, "WEBP"):
		return "image/webp"
	case startsWithAscii(b, 0, "BM") && isBmp(b):
		return "image/bmp"
	default:
		return ""
	}
}

func startsWith(b, prefix []byte) bool {
	if len(b) < len(prefix) {
		return false
	}
	for i := range prefix {
		if b[i] != prefix[i] {
			return false
		}
	}
	return true
}

func startsWithAscii(b []byte, offset int, text string) bool {
	if len(b) < offset+len(text) {
		return false
	}
	for i := 0; i < len(text); i++ {
		if b[offset+i] != text[i] {
			return false
		}
	}
	return true
}

func readUint32BE(b []byte, offset int) uint32 {
	if offset+4 > len(b) {
		return 0
	}
	return uint32(b[offset])<<24 | uint32(b[offset+1])<<16 | uint32(b[offset+2])<<8 | uint32(b[offset+3])
}

func readUint32LE(b []byte, offset int) uint32 {
	if offset+4 > len(b) {
		return 0
	}
	return uint32(b[offset]) | uint32(b[offset+1])<<8 | uint32(b[offset+2])<<16 | uint32(b[offset+3])<<24
}

func readUint16LE(b []byte, offset int) uint16 {
	if offset+2 > len(b) {
		return 0
	}
	return uint16(b[offset]) | uint16(b[offset+1])<<8
}

func isPng(b []byte) bool {
	return len(b) >= 16 && readUint32BE(b, 8) == 13 && startsWithAscii(b, 12, "IHDR")
}

func isAnimatedPng(b []byte) bool {
	offset := 8
	for offset+8 <= len(b) {
		chunkLength := readUint32BE(b, offset)
		if startsWithAscii(b, offset+4, "acTL") {
			return true
		}
		if startsWithAscii(b, offset+4, "IDAT") {
			return false
		}
		next := offset + 8 + int(chunkLength) + 4
		if next <= offset || next > len(b) {
			return false
		}
		offset = next
	}
	return false
}

func isBmp(b []byte) bool {
	if len(b) < 26 {
		return false
	}
	declaredFileSize := readUint32LE(b, 2)
	pixelDataOffset := readUint32LE(b, 10)
	dibHeaderSize := readUint32LE(b, 14)
	if declaredFileSize != 0 && declaredFileSize < 26 {
		return false
	}
	if pixelDataOffset < 14+dibHeaderSize {
		return false
	}
	if declaredFileSize != 0 && pixelDataOffset >= declaredFileSize {
		return false
	}

	var colorPlanes, bitsPerPixel uint16
	if dibHeaderSize == 12 {
		colorPlanes = readUint16LE(b, 22)
		bitsPerPixel = readUint16LE(b, 24)
	} else if dibHeaderSize >= 40 && dibHeaderSize <= 124 {
		if len(b) < 30 {
			return false
		}
		colorPlanes = readUint16LE(b, 26)
		bitsPerPixel = readUint16LE(b, 28)
	} else {
		return false
	}
	if colorPlanes != 1 {
		return false
	}
	switch bitsPerPixel {
	case 1, 4, 8, 16, 24, 32:
		return true
	}
	return false
}
