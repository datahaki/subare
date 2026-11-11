// code by jph
package ch.alpine.subare.net;

import ch.alpine.tensor.Rational;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.VectorQ;
import ch.alpine.tensor.nrm.Vector2NormSquared;
import ch.alpine.tensor.red.Mean;

/** Mean Squared Error */
public record MSELoss(Scalar loss, Tensor gradOutput) implements NetError {
  public static NetError of(Tensor y_pred, Tensor target) {
    return of(y_pred.subtract(target));
  }

  private static NetError of(Tensor error) {
    if (VectorQ.of(error))
      return new MSELoss( //
          Vector2NormSquared.of(error), //
          error.multiply(RealScalar.TWO));
    Scalar loss = Mean.ofVector(Tensor.of(error.stream().map(Vector2NormSquared::of)));
    int batch_size = error.length();
    Tensor gradOutput = error.multiply(Rational.of(2, batch_size));
    return new MSELoss(loss, gradOutput);
  }
}
