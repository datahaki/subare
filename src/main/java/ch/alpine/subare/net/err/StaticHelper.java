package ch.alpine.subare.net.err;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.api.ScalarUnaryOperator;
import ch.alpine.tensor.red.Max;
import ch.alpine.tensor.sca.Sign;
import ch.alpine.tensor.sca.exp.Log;

enum StaticHelper {
  ;
  static final ScalarUnaryOperator MAX = Max.function(RealScalar.of(1e-15));

  public static Scalar safeLog(Scalar value) {
    return Log.FUNCTION.apply(MAX.apply(Sign.requirePositiveOrZero(value)));
  }
}
