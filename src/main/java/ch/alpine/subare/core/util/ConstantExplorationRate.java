// code by fluric
package ch.alpine.subare.core.util;

import ch.alpine.subare.core.StateActionCounter;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

/** learning rate of alpha except in first update of state-action pair
 * for which the learning rate equals 1 in the case of warmStart. */
public class ConstantExplorationRate implements ExplorationRate {
  public static ExplorationRate of(Scalar epsilon) {
    return new ConstantExplorationRate(epsilon);
  }

  /** @param epsilon
   * @return constant learning rate with factor alpha */
  public static ExplorationRate of(Number epsilon) {
    return of(RealScalar.of(epsilon));
  }

  // ---
  private Scalar epsilon;

  private ConstantExplorationRate(Scalar epsilon) {
    this.epsilon = epsilon;
  }

  @Override
  public Scalar epsilon(Tensor state, StateActionCounter stateActionCounter) {
    return epsilon;
  }
}
