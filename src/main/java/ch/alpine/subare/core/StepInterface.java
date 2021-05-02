// code by jph
package ch.alpine.subare.core;

import ch.alpine.subare.core.adapter.StepAdapter;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

/** class provides the four entries (s, a, r, s')
 * 
 * instances of implementations are required to be immutable
 * Example: {@link StepAdapter} */
public interface StepInterface {
  /** @return previous state */
  Tensor prevState();

  /** @return action that was taken to reach next state */
  Tensor action();

  /** @return reward */
  Scalar reward();

  /** @return next state */
  Tensor nextState();
}
