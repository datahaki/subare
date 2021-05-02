// code by jph and fluric
package ch.ethz.idsc.subare.core.util;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.ethz.idsc.subare.core.StateActionCounter;

@FunctionalInterface
public interface ExplorationRate {
  /** @param state
   * @param stateActionCounter
   * @return exploration rate for given state-action pair */
  Scalar epsilon(Tensor state, StateActionCounter stateActionCounter);
}
