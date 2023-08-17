// code by jph
package ch.alpine.subare.core.api;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

public interface TransitionInterface {
  /** @param state
   * @param action
   * @return all states that are a possible result of taking action in given state */
  Tensor transitions(Tensor state, Tensor action);

  /** @param state
   * @param action
   * @param next
   * @return probability to reach next as a result of taking action in given state */
  Scalar transitionProbability(Tensor state, Tensor action, Tensor next);
}
