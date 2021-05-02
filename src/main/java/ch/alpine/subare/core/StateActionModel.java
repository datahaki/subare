// code by jph
package ch.alpine.subare.core;

import ch.alpine.tensor.Tensor;

public interface StateActionModel {
  /** @return all states */
  Tensor states();

  /** for a terminal state, the returned actions(state) should have length() == 1
   * 
   * @param state
   * @return all actions possible to execute from given state */
  Tensor actions(Tensor state);
}
