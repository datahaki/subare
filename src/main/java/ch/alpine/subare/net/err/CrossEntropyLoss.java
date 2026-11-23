// code by jph
package ch.alpine.subare.net.err;

import java.util.List;

import ch.alpine.subare.net.LossFunction;
import ch.alpine.subare.net.NetLoss;
import ch.alpine.tensor.Rational;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.api.ScalarUnaryOperator;
import ch.alpine.tensor.ext.Int;
import ch.alpine.tensor.ext.Integers;
import ch.alpine.tensor.red.Max;
import ch.alpine.tensor.red.Mean;
import ch.alpine.tensor.sca.Sign;
import ch.alpine.tensor.sca.exp.Log;

public enum CrossEntropyLoss implements LossFunction {
  INSTANCE;

  static final ScalarUnaryOperator MAX = Max.function(RealScalar.of(1e-15));

  /** @param prediction non-negative probabilities
   * @param target is a scalar
   * vector of length 1, or matrix of dimensions n x 1
   * of integer values between 0 and interCache.length
   * @return */
  @Override
  public NetLoss compute(Tensor prediction, Tensor target) {
    int k = Scalars.intValueExact((Scalar) target);
    Scalar loss = safeLog(prediction.Get(k)).negate();
    Tensor gradOutput = prediction.copy();
    gradOutput.set(RealScalar.ONE.negate()::add, k);
    return new NetLoss(loss, gradOutput);
  }

  /** @param prediction matrix where rows are non-negative probabilities
   * @param target vector of integers
   * @return */
  @Override
  public NetLoss computeBatch(Tensor prediction, Tensor target) {
    int batch_size = Integers.requireEquals(prediction.length(), target.length());
    Int myI = new Int();
    List<NetLoss> list = prediction.stream() //
        .map(pred -> compute(pred, target.Get(myI.getAndIncrement()))) //
        .toList();
    Scalar loss = (Scalar) Mean.of(list.stream().map(NetLoss::loss));
    Scalar factor = Rational.of(1, batch_size);
    Tensor gradOutput = Tensor.of(list.stream() //
        .map(NetLoss::gradOutput) //
        .map(grad -> grad.multiply(factor)), batch_size);
    return new NetLoss(loss, gradOutput);
  }

  public static Scalar safeLog(Scalar value) {
    return Log.FUNCTION.apply(MAX.apply(Sign.requirePositiveOrZero(value)));
  }
}
