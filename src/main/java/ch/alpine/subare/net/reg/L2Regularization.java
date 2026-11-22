// code by jph
package ch.alpine.subare.net.reg;

import java.io.Serializable;

import ch.alpine.tensor.Rational;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Flatten;
import ch.alpine.tensor.io.MathematicaFormat;
import ch.alpine.tensor.sca.AbsSquared;
import ch.alpine.tensor.sca.Sign;

/** possibly in gradient */
public record L2Regularization(Scalar lambda) implements NetRegularization, Serializable {
  public L2Regularization {
    Sign.requirePositiveOrZero(lambda);
  }

  @Override
  public Scalar penalty(boolean decayInLoss, Tensor weights) {
    return decayInLoss //
        ? Flatten.scalars(weights) //
            .map(AbsSquared.FUNCTION) //
            .reduce(Scalar::add) //
            .orElseThrow() //
            .multiply(lambda) //
            .multiply(Rational.HALF)
        : RealScalar.ZERO;
  }

  @Override
  public Tensor gradient(boolean decayInLoss, Tensor weights) {
    return decayInLoss //
        ? weights.multiply(lambda)
        : weights.maps(Scalar::zero);
  }

  @Override
  public Tensor sneakDecay(boolean decayInLoss, Tensor weights) {
    return decayInLoss //
        ? weights.maps(Scalar::zero)
        : weights.multiply(lambda);
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("L2", lambda);
  }
}
