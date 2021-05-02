// code by jph
package ch.ethz.idsc.subare.core;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

@FunctionalInterface
public interface RewardInterface {
  /** the reward function is not necessarily deterministic
   * 
   * @param state
   * @param action
   * @param next
   * @return reward may vary even for invariant input */
  Scalar reward(Tensor state, Tensor action, Tensor next);
}
