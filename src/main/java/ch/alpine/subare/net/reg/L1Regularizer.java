// code by jph
package ch.alpine.subare.net.reg;

import java.io.Serializable;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Flatten;
import ch.alpine.tensor.sca.Abs;
import ch.alpine.tensor.sca.Sign;

public record L1Regularizer(Scalar lambda) implements NetRegularizer, Serializable {
  @Override
  public Scalar penalty(Tensor weights) {
    return Flatten.scalars(weights) //
        .map(Abs.FUNCTION) //
        .reduce(Scalar::add) //
        .orElseThrow().multiply(lambda);
  }

  @Override
  public Tensor gradient(Tensor weights) {
    return weights.maps(Sign.FUNCTION).multiply(lambda);
  }
}
