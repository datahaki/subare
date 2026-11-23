// code by jph
package ch.alpine.subare.net.err;

import java.io.Serializable;

import ch.alpine.subare.net.NetLoss;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.nrm.Vector2NormSquared;

/** Loss == Squared Error divided by batch_size */
public record SquaredErrorLoss(Scalar loss, Tensor gradOutput) implements NetLoss, Serializable {
  public static NetLoss of(Tensor y_pred, Tensor target) {
    return of(y_pred.subtract(target));
  }

  private static NetLoss of(Tensor error) {
    Tensor gradOutput = error.add(error); // this is where *2 is
    return new SquaredErrorLoss( //
        Vector2NormSquared.of(error), //
        gradOutput);
  }
}
