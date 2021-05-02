// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.stream.Stream;

import ch.alpine.tensor.Tensor;

public interface DiscreteValueFunction {
  /** @return */
  Tensor keys();

  /** @return unmodifiable vector of (state)-, or (state, action)-values */
  Tensor values();

  DiscreteValueFunction create(Stream<? extends Tensor> stream);
}
