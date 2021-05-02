// code by jph
package ch.alpine.subare.core;

import ch.alpine.tensor.Tensor;

@FunctionalInterface
public interface MoveInterface {
  /** the move function is not necessarily deterministic, i.e.
   * two consecutive calls may return different values
   * 
   * @param state
   * @param action
   * @return new state as consequence of given state and action */
  Tensor move(Tensor state, Tensor action);
}
