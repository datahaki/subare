// code by fluric
package ch.alpine.subare.core.util;

import ch.alpine.subare.core.StateActionCounter;
import ch.alpine.subare.core.StepInterface;
import ch.alpine.tensor.Scalar;

/** THE USE OF THIS CLASS IS NOT RECOMMENDED BECAUSE THE
 * UPDATE IS BIASED TOWARDS AN UNWARRANTED INITIAL VALUE */
public class StrictConstantLearningRate implements LearningRate {
  private final Scalar alpha;

  public StrictConstantLearningRate(Scalar alpha) {
    this.alpha = alpha;
  }

  @Override // from LearningRate
  public Scalar alpha(StepInterface stepInterface, StateActionCounter stateActionCounter) {
    return alpha;
  }
}
