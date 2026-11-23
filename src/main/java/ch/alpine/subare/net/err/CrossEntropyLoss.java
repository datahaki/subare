// code by jph
package ch.alpine.subare.net.err;

import java.io.Serializable;

import ch.alpine.subare.net.NetLoss;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.api.ScalarUnaryOperator;
import ch.alpine.tensor.red.Max;

/** Loss == Cross Entropy divided by batch_size */
public record CrossEntropyLoss(Scalar loss, Tensor gradOutput) implements NetLoss, Serializable {
  static final ScalarUnaryOperator MAX = Max.function(RealScalar.of(1e-15));

  /** @param interCache non-negative probabilities
   * @param target
   * vector of length 1, or matrix of dimensions n x 1
   * of integer values between 0 and interCache.length
   * @return */
  public static NetLoss of(Tensor interCache, Tensor target) {
    int k = Scalars.intValueExact((Scalar) target);
    Scalar loss = StaticHelper.safeLog(interCache.Get(k)).negate();
    Tensor error = interCache.copy();
    error.set(RealScalar.ONE.negate()::add, k);
    return new CrossEntropyLoss(loss, error);
  }
}
