// code by jph
package ch.alpine.subare.core.util.gfx;

import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.api.ScalarUnaryOperator;
import ch.alpine.tensor.chq.FiniteScalarQ;
import ch.alpine.tensor.sca.Clips;

/** clips scalars to interval [0, 1] excluding scalars that do not satisfy {@link FiniteScalarQ}
 * such as {@link DoubleScalar#POSITIVE_INFINITY} and {@link DoubleScalar#INDETERMINATE} */
/* package */ enum UnitClip implements ScalarUnaryOperator {
  FUNCTION;

  @Override
  public Scalar apply(Scalar scalar) {
    return FiniteScalarQ.of(scalar) //
        ? Clips.unit().apply(scalar)
        : scalar;
  }
}
