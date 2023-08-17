// code by fluric
package ch.alpine.subare.util;

import java.io.Serializable;

import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.red.Max;
import ch.alpine.tensor.sca.Clips;
import ch.alpine.tensor.sca.Sign;

/** using formula: epsilon = Max(minimum, maximum-(maximum-minimum)*N/decayInterval) */
public class LinearExplorationRate implements ExplorationRate, Serializable {
  /** good values could be: decayInterval=1000, maximum=0.5, minimum=0.01, strongly depends on the problem
   * 
   * @param decayInterval
   * @param maximum
   * @param minimum
   * @return */
  public static ExplorationRate of(Scalar decayInterval, Scalar maximum, Scalar minimum) {
    return new LinearExplorationRate(decayInterval, maximum, minimum);
  }

  public static ExplorationRate of(Number decayInterval, Number maximum, Number minimum) {
    return of(RealScalar.of(decayInterval), RealScalar.of(maximum), RealScalar.of(minimum));
  }

  // ---
  private final Scalar decayInterval;
  private final Scalar minimum;
  private final Scalar maximum;

  // TODO SUBARE input parameter type one Clip instead of two scalars
  private LinearExplorationRate(Scalar decayInterval, Scalar maximum, Scalar minimum) {
    this.decayInterval = Sign.requirePositive(decayInterval);
    Clips.interval(minimum, maximum);
    this.minimum = Clips.unit().requireInside(minimum);
    this.maximum = Clips.unit().requireInside(maximum);
  }

  @Override // from ExplorationRate
  public final Scalar epsilon(Tensor state, StateActionCounter sac) {
    return epsilon(sac.stateCount(state));
  }

  final Scalar epsilon(Scalar stateCount) {
    // TODO SUBARE explore
    Scalar decayedValue = maximum.subtract(maximum.subtract(minimum).multiply(stateCount).divide(decayInterval));
    return Max.of(minimum, decayedValue);
  }
}
