// code by jph
package ch.alpine.subare.mod;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

public interface DiscreteModel {
  /** @return all states, elements are unique */
  Tensor states();

  /** for a terminal state, the returned actions(state) should have length() == 1
   * 
   * @param state
   * @return all actions possible to execute from given state */
  Tensor actions(Tensor state);

  /** @return discount factor in the interval [0, 1] */
  Scalar gamma();
}
