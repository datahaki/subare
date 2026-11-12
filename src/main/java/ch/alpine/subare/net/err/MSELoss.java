// code by jph
package ch.alpine.subare.net.err;

import ch.alpine.tensor.Rational;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.VectorQ;
import ch.alpine.tensor.nrm.Vector2NormSquared;
import ch.alpine.tensor.red.Mean;

/** Mean Squared Error */
public record MSELoss(Scalar loss, Tensor gradOutput) implements NetError {
  public static NetError of(Tensor y_pred, Tensor target) {
    // IO.println(y_pred);
    // IO.println(target);
    // IO.println(Dimensions.of(y_pred));
    // IO.println(Dimensions.of(target));
    return of(y_pred.subtract(target));
  }

  private static NetError of(Tensor error) {
    if (VectorQ.of(error)) {
      Tensor gradOutput = error.add(error);
      return new MSELoss( //
          Vector2NormSquared.of(error), //
          gradOutput);
    }
    // mean loss
    Scalar loss = Mean.ofVector(Tensor.of(error.stream().map(Vector2NormSquared::of)));
    int batch_size = error.length();
    Tensor gradOutput = error.multiply(Rational.of(2, batch_size));
    return new MSELoss(loss, gradOutput);
  }
}
