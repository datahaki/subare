// code by jph
package ch.alpine.subare.net.reg;

import java.io.Serializable;

import ch.alpine.subare.net.api.NetRegularization;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Flatten;
import ch.alpine.tensor.io.MathematicaFormat;
import ch.alpine.tensor.sca.Abs;
import ch.alpine.tensor.sca.Sign;

public record L1Regularization(Scalar lambda) implements NetRegularization, Serializable {
  public L1Regularization {
    Sign.requirePositiveOrZero(lambda);
  }

  @Override
  public Scalar penalty(boolean weightDecay, Tensor weights) {
    return Flatten.scalars(weights) //
        .map(Abs.FUNCTION) //
        .reduce(Scalar::add) //
        .orElseThrow().multiply(lambda);
  }

  @Override
  public Tensor gradient(boolean weightDecay, Tensor weights) {
    return weights.maps(Sign.FUNCTION).multiply(lambda);
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("L1", lambda);
  }
}
