// code by jph
package ch.alpine.subare.net.reg;

import java.io.Serializable;

import ch.alpine.tensor.Rational;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Flatten;
import ch.alpine.tensor.io.MathematicaFormat;
import ch.alpine.tensor.sca.AbsSquared;

public record L2Regularizer(Scalar lambda) implements NetRegularizer, Serializable {
  @Override
  public Scalar penalty(Tensor weights) {
    return Flatten.scalars(weights) //
        .map(AbsSquared.FUNCTION) //
        .reduce(Scalar::add) //
        .orElseThrow() //
        .multiply(lambda) //
        .multiply(Rational.HALF);
  }

  @Override
  public Tensor gradient(Tensor weights, Scalar eta) {
    return weights.multiply(lambda.multiply(eta));
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("L2", lambda);
  }
}
