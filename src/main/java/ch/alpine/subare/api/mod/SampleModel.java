// code by jph
package ch.alpine.subare.api.mod;

import ch.alpine.tensor.Tensor;

/** name of class is motivated by box on p.169 */
public interface SampleModel extends RewardInterface {
  /** the move function is not necessarily deterministic, i.e.
   * two consecutive calls may return different values
   * 
   * @param state
   * @param action
   * @return new state as consequence of given state and action */
  Tensor move(Tensor state, Tensor action);
}
