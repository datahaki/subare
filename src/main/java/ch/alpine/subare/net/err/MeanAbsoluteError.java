// code by gemini
package ch.alpine.subare.net.err;

import ch.alpine.subare.net.LossFunction;
import ch.alpine.subare.net.NetLoss;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Flatten;
import ch.alpine.tensor.alg.Numel;
import ch.alpine.tensor.red.Mean;
import ch.alpine.tensor.sca.Abs;
import ch.alpine.tensor.sca.Sign;

/** MeanAbsoluteError accepts network output of arbitrary structure */
public enum MeanAbsoluteError implements LossFunction {
  INSTANCE;

  @Override
  public NetLoss compute(Tensor prediction, Tensor target) {
    Tensor error = prediction.subtract(target);
    return new NetLoss( //
        (Scalar) Mean.of(Flatten.scalars(error).map(Abs.FUNCTION)), //
        error.maps(Sign.FUNCTION).divide(RealScalar.of(Numel.of(error))));
  }

  @Override
  public NetLoss computeBatch(Tensor prediction, Tensor target) {
    // intenional
    return compute(prediction, target);
  }

  @Override
  public String toString() {
    return "MeanAbsoluteError";
  }
}
