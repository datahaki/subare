// code by jph
package ch.alpine.subare.net.err;

import ch.alpine.subare.net.LossFunction;
import ch.alpine.subare.net.NetLoss;
import ch.alpine.tensor.Rational;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Slash;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.nrm.Vector2NormSquared;
import ch.alpine.tensor.red.Mean;

/** Loss == Squared Error divided by batch_size */
public enum SquaredErrorLoss implements LossFunction {
  INSTANCE;

  @Override
  public NetLoss compute(Tensor prediction, Tensor target) {
    Tensor error = prediction.subtract(target);
    Tensor gradOutput = error.add(error); // this is where *2 is
    return new NetLoss( //
        Vector2NormSquared.of(error), //
        gradOutput);
  }

  @Override
  public NetLoss computeBatch(Tensor y_pred, Tensor target) {
    Tensor error = y_pred.subtract(target);
    Scalar loss = Mean.ofVector(Slash.of(error, Vector2NormSquared::of));
    int batch_size = error.length();
    Tensor gradOutput = error.multiply(Rational.of(2, batch_size));
    return new NetLoss(loss, gradOutput);
  }
}
