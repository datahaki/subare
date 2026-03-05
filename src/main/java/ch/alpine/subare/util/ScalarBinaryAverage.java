// code by jph
package ch.alpine.subare.util;

import ch.alpine.tensor.Scalar;

public enum ScalarBinaryAverage {
  INSTANCE;

  public Scalar split(Scalar p, Scalar q, Scalar scalar) {
    Scalar shift = q.subtract(p).multiply(scalar);
    return scalar.one().equals(scalar) //
        ? q
        : p.add(shift);
  }
}
