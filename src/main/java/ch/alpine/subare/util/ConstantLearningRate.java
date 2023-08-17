// code by jph and fluric
package ch.alpine.subare.util;

import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.api.StepRecord;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;

/** learning rate of alpha except in first update of state-action pair
 * for which the learning rate equals 1 in the case of warmStart. */
public record ConstantLearningRate(Scalar alpha) implements LearningRate {
  /** @param alpha
   * @return constant learning rate with factor alpha */
  public static LearningRate of(Scalar alpha) {
    return new ConstantLearningRate(alpha);
  }

  /** @return constant learning rate with factor 1.0,
   * that means the updates have numeric precision */
  public static LearningRate one() {
    return of(RealScalar.of(1.0));
  }

  /** @return constant learning rate with exact factor 1,
   * that means the precision in the updates is preserved */
  public static LearningRate one_exact() {
    return of(RealScalar.ONE);
  }

  // ---
  @Override
  public Scalar alpha(StepRecord stepInterface, StateActionCounter stateActionCounter) {
    return stateActionCounter.isEncountered(StateAction.key(stepInterface)) ? alpha : RealScalar.ONE;
  }
}
