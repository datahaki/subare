// code by jph and fluric
package ch.alpine.subare.util;

import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.api.StepRecord;
import ch.alpine.tensor.Scalar;

@FunctionalInterface
public interface LearningRate {
  /** successive calls to the function give the same result.
   * 
   * the first call to the function should return numerical value == 1
   * to prevent initialization bias.
   * 
   * @param {@link StepRecord}
   * @param {@link StateActionCounter}
   * @return learning rate for given state-action pair */
  Scalar alpha(StepRecord stepInterface, StateActionCounter stateActionCounter);
}
