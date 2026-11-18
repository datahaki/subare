// code by jph
package ch.alpine.subare.net.reg;

import java.io.Serializable;

import ch.alpine.subare.net.api.NetRegularization;
import ch.alpine.tensor.Rational;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Flatten;
import ch.alpine.tensor.io.MathematicaFormat;
import ch.alpine.tensor.sca.AbsSquared;
import ch.alpine.tensor.sca.Sign;

public record L2Regularization(Scalar lambda) implements NetRegularization, Serializable {
  public L2Regularization {
    Sign.requirePositiveOrZero(lambda);
  }

  @Override
  public Scalar penalty(boolean weightDecay, Tensor weights) {
    return weightDecay //
        ? Flatten.scalars(weights) //
            .map(AbsSquared.FUNCTION) //
            .reduce(Scalar::add) //
            .orElseThrow() //
            .multiply(lambda) //
            .multiply(Rational.HALF)
        : RealScalar.ZERO;
  }

  @Override
  public Tensor gradient(boolean weightDecay, Tensor weights) {
    return weightDecay //
        ? weights.multiply(lambda)
        : weights.maps(Scalar::zero);
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("L2", lambda);
  }
}
