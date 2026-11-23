// code by jph
package ch.alpine.subare.net.err;

import java.io.Serializable;

import ch.alpine.subare.net.NetLoss;
import ch.alpine.tensor.Rational;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Slash;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.nrm.Vector2NormSquared;
import ch.alpine.tensor.red.Mean;

/** Loss == Squared Error divided by batch_size */
public record SquaredErrorBatchLoss(Scalar loss, Tensor gradOutput) implements NetLoss, Serializable {
  public static NetLoss of(Tensor y_pred, Tensor target) {
    return of(y_pred.subtract(target));
  }

  private static NetLoss of(Tensor error) {
    Scalar loss = Mean.ofVector(Slash.of(error, Vector2NormSquared::of));
    int batch_size = error.length();
    Tensor gradOutput = error.multiply(Rational.of(2, batch_size));
    return new SquaredErrorBatchLoss(loss, gradOutput);
  }
}
