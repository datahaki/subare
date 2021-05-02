// code by jph
package ch.ethz.idsc.subare.core.util.gfx;

import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.NumberQ;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.api.ScalarUnaryOperator;
import ch.alpine.tensor.sca.Clips;

/** clips scalars to interval [0, 1] excluding scalars that do not satisfy {@link NumberQ}
 * such as {@link DoubleScalar#POSITIVE_INFINITY} and {@link DoubleScalar#INDETERMINATE} */
/* package */ enum UnitClip implements ScalarUnaryOperator {
  FUNCTION;

  @Override
  public Scalar apply(Scalar scalar) {
    return NumberQ.of(scalar) //
        ? Clips.unit().apply(scalar)
        : scalar;
  }

  @SuppressWarnings("unchecked")
  public static <T extends Tensor> T of(T tensor) {
    return (T) tensor.map(FUNCTION);
  }
}
