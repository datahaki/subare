// code by jph, fluric
package ch.alpine.subare.rate;

import ch.alpine.subare.api.StepRecord;
import ch.alpine.subare.pol.StateActionCounter;
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
  Scalar alpha(StepRecord stepRecord, StateActionCounter stateActionCounter);
}
