// code by fluric
package ch.alpine.subare.util;

import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.api.StepRecord;
import ch.alpine.tensor.Scalar;

/** THE USE OF THIS CLASS IS NOT RECOMMENDED BECAUSE THE
 * UPDATE IS BIASED TOWARDS AN UNWARRANTED INITIAL VALUE */
public record StrictConstantLearningRate(Scalar alpha) implements LearningRate {
  @Override // from LearningRate
  public Scalar alpha(StepRecord stepInterface, StateActionCounter stateActionCounter) {
    return alpha;
  }
}
