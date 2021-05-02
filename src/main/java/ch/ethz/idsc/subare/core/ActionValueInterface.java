// code by jph
package ch.ethz.idsc.subare.core;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

public interface ActionValueInterface extends TransitionInterface {
  /** @param state
   * @param action
   * @return expected reward when action is taken in state */
  Scalar expectedReward(Tensor state, Tensor action);
}
