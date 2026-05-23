// code by chatgpt
package ch.alpine.subare.math;

enum Float16 {
  ;
  public static short floatToHalf(float f) {
    int bits = Float.floatToIntBits(f);
    int sign = (bits >>> 16) & 0x8000;
    int val = (bits & 0x7fffffff) + 0x1000;
    if (val >= 0x47800000) {
      if ((bits & 0x7fffffff) >= 0x47800000) {
        if (val < 0x7f800000)
          return (short) (sign | 0x7c00); // Inf
        return (short) (sign | 0x7c00 | ((bits & 0x007fffff) >>> 13)); // NaN
      }
      return (short) (sign | 0x7bff); // max
    }
    if (val >= 0x38800000)
      return (short) (sign | ((val - 0x38000000) >>> 13));
    if (val < 0x33000000)
      return (short) sign;
    val = (bits & 0x7fffffff) >>> 23;
    return (short) (sign | ((((bits & 0x7fffff) | 0x800000) + (0x800000 >>> (val - 102))) >>> (126 - val)));
  }

  public static float halfToFloat(short h) {
    int sign = (h & 0x8000) << 16;
    int exp = (h & 0x7c00) >> 10;
    int mant = h & 0x03ff;
    int bits;
    if (exp == 0) {
      if (mant == 0) {
        bits = sign;
      } else {
        while ((mant & 0x0400) == 0) {
          mant <<= 1;
          exp--;
        }
        exp++;
        mant &= ~0x0400;
        bits = sign | ((exp + 112) << 23) | (mant << 13);
      }
    } else if (exp == 31) {
      bits = sign | 0x7f800000 | (mant << 13);
    } else {
      bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    return Float.intBitsToFloat(bits);
  }
}
