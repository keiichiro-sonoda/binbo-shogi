"""Task 149: versioned HalfKP integer header; legacy BNNQ bytes stay unchanged."""
import struct

MAGIC = b"BNNQ"
MAGIC_V2 = b"BNQ2"
HALFKP = 1


def header(total, fd, h1, h2, fv_scale=16, weight_scale_bits=6):
    """Task 149: BNQ2 adds H2 and a feature id; tensor order is unchanged."""
    if min(total, fd, h1, h2) < 1:
        raise ValueError("quantized dimensions must be positive")
    if h1 == h2:
        return MAGIC + struct.pack("<IIIiI", total, fd, h1, fv_scale, weight_scale_bits)
    return MAGIC_V2 + struct.pack("<IIIIiII", total, fd, h1, h2,
                                 fv_scale, weight_scale_bits, HALFKP)


def read_header(buf):
    """Task 149: reject unknown features/versions before reading tensors."""
    if buf[:4] == MAGIC:
        total, fd, h1, fv, bits = struct.unpack_from("<IIIiI", buf, 4)
        h2, offset = h1, 24
    elif buf[:4] == MAGIC_V2:
        total, fd, h1, h2, fv, bits, feature = struct.unpack_from("<IIIIiII", buf, 4)
        if feature != HALFKP:
            raise ValueError(f"unsupported BNQ2 feature id: {feature}")
        offset = 32
    else:
        raise ValueError(f"unsupported quantized magic: {buf[:4]!r}")
    if min(total, fd, h1, h2) < 1 or (fv, bits) != (16, 6):
        raise ValueError("unsupported quantized dimensions/scales")
    return dict(total=total, fd=fd, hidden=h1, hidden2=h2, fv_scale=fv,
                weight_scale_bits=bits), offset
