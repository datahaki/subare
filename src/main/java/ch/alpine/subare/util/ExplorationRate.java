// code by jph and fluric
package ch.alpine.subare.util;

import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

@FunctionalInterface
public interface ExplorationRate {
  /** @param state
   * @param stateActionCounter
   * @return exploration rate for given state-action pair */
  Scalar epsilon(Tensor state, StateActionCounter stateActionCounter);
}
