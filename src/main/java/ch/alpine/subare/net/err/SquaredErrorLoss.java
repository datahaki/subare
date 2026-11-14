// code by jph
package ch.alpine.subare.net.err;

import java.io.Serializable;

import ch.alpine.tensor.Rational;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.VectorQ;
import ch.alpine.tensor.nrm.Vector2NormSquared;
import ch.alpine.tensor.red.Mean;

/** Loss == Squared Error divided by batch_size */
public record SquaredErrorLoss(Scalar loss, Tensor gradOutput) implements NetLoss, Serializable {
  public static NetLoss of(Tensor y_pred, Tensor target) {
    return of(y_pred.subtract(target));
  }

  private static NetLoss of(Tensor error) {
    if (VectorQ.of(error)) {
      Tensor gradOutput = error.add(error);
      return new SquaredErrorLoss( //
          Vector2NormSquared.of(error), //
          gradOutput);
    }
    // mean loss
    Scalar loss = Mean.ofVector(Tensor.of(error.stream().map(Vector2NormSquared::of)));
    int batch_size = error.length();
    Tensor gradOutput = error.multiply(Rational.of(2, batch_size));
    return new SquaredErrorLoss(loss, gradOutput);
  }
}
