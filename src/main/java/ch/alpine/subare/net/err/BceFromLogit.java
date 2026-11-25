// adapted from gemini
package ch.alpine.subare.net.err;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.api.ScalarBinaryOperator;
import ch.alpine.tensor.sca.Abs;
import ch.alpine.tensor.sca.Ramp;
import ch.alpine.tensor.sca.exp.Exp;
import ch.alpine.tensor.sca.exp.Log;

public enum BceFromLogit implements ScalarBinaryOperator {
  FUNCTION;

  @Override
  public Scalar apply(Scalar z, Scalar target) { // target is either 0 or 1
    Scalar rampZ = Ramp.FUNCTION.apply(z);
    Scalar negAbsZ = Exp.FUNCTION.apply(Abs.FUNCTION.apply(z).negate());
    Scalar logTerm = Log.FUNCTION.apply(RealScalar.ONE.add(negAbsZ));
    Scalar zTimesY = z.multiply(target);
    return rampZ.subtract(zTimesY).add(logTerm);
  }
}
