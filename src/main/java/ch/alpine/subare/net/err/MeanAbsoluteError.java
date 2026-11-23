// code by gemini
package ch.alpine.subare.net.err;

import ch.alpine.subare.net.LossFunction;
import ch.alpine.subare.net.NetLoss;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Flatten;
import ch.alpine.tensor.red.Total;
import ch.alpine.tensor.sca.Abs;
import ch.alpine.tensor.sca.Sign;

// TODO clean this up
public enum MeanAbsoluteError implements LossFunction {
  INSTANCE;

  @Override
  public NetLoss compute(Tensor prediction, Tensor target) {
    Tensor error = prediction.subtract(target);
    Scalar loss = (Scalar) Total.of(error.maps(Abs.FUNCTION)).divide(RealScalar.of(error.length()));
    Tensor gradient = error.maps(Sign.FUNCTION).divide(RealScalar.of(error.length()));
    return new NetLoss(loss, gradient);
  }

  @Override
  public NetLoss computeBatch(Tensor prediction, Tensor target) {
    Tensor error = prediction.subtract(target);
    // Total number of elements across the batch and feature dimensions
    int totalElements = Flatten.of(error).length();
    Scalar loss = (Scalar) Total.of(Flatten.of(error).maps(Abs.FUNCTION)).divide(RealScalar.of(totalElements));
    Tensor gradient = error.maps(Sign.FUNCTION).divide(RealScalar.of(totalElements));
    return new NetLoss(loss, gradient);
  }
}
