// code by jph
package ch.alpine.subare.net.err;

import ch.alpine.subare.net.LossFunction;
import ch.alpine.subare.net.NetLoss;
import ch.alpine.tensor.Rational;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.nrm.FrobeniusNorm;

/** Loss == Squared Error divided by batch_size */
public enum SquaredErrorLoss implements LossFunction {
  INSTANCE;

  @Override
  public NetLoss compute(Tensor prediction, Tensor target) {
    Tensor error = prediction.subtract(target);
    return new NetLoss( //
        FrobeniusNorm.of(error), //
        error.add(error));
  }

  @Override
  public NetLoss computeBatch(Tensor prediction, Tensor target) {
    Tensor error = prediction.subtract(target);
    int batch_size = error.length();
    return new NetLoss( //
        FrobeniusNorm.of(error).divide(RealScalar.of(batch_size)), //
        error.multiply(Rational.of(2, batch_size)));
  }

  @Override
  public String toString() {
    return "SquaredErrorLoss";
  }
}
