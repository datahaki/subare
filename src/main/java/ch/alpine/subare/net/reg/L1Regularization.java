// code by jph
package ch.alpine.subare.net.reg;

import java.io.Serializable;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Flatten;
import ch.alpine.tensor.io.MathematicaFormat;
import ch.alpine.tensor.sca.Abs;
import ch.alpine.tensor.sca.Sign;

/** always in gradient */
public record L1Regularization(Scalar lambda) implements NetRegularization, Serializable {
  public L1Regularization {
    Sign.requirePositiveOrZero(lambda);
  }

  @Override // always part of loss
  public Scalar penalty(boolean decayInLoss, Tensor weights) {
    return Flatten.scalars(weights) //
        .map(Abs.FUNCTION) //
        .reduce(Scalar::add) //
        .orElseThrow() //
        .multiply(lambda);
  }

  @Override // always part of gradient
  public Tensor gradient(boolean decayInLoss, Tensor weights) {
    return weights.maps(Sign.FUNCTION).multiply(lambda);
  }

  @Override // never sneak past gradient
  public Tensor sneakDecay(boolean decayInLoss, Tensor weights) {
    return weights.maps(Scalar::zero);
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("L1", lambda);
  }
}
