// code by fluric
package ch.ethz.idsc.subare.core.util;

import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.red.Min;
import ch.alpine.tensor.sca.Power;
import ch.alpine.tensor.sca.Sign;
import ch.ethz.idsc.subare.core.StateActionCounter;

/** using formula: epsilon = factor*(1/(1+N))^(exponent), N is the number of state visits
 * good values are: factor = 0.5, exponent = 0.5, depends strongly on problem */
public class DecayedExplorationRate implements ExplorationRate {
  public static DecayedExplorationRate of(double factor, double exponent) {
    return new DecayedExplorationRate(RealScalar.of(factor), RealScalar.of(exponent));
  }

  // ---
  private final Scalar factor;
  private final Scalar exponent;
  /** lookup table to speed up computation */
  private final Tensor MEMO = Tensors.vector(1.0); // index == 0 => learning rate == 1

  /* package */ DecayedExplorationRate(Scalar factor, Scalar exponent) {
    this.factor = Sign.requirePositiveOrZero(factor);
    this.exponent = exponent;
  }

  @Override // from ExplorationRate
  public synchronized final Scalar epsilon(Tensor state, StateActionCounter stateActionCounter) {
    int index = Scalars.intValueExact(stateActionCounter.stateCount(state));
    while (MEMO.length() <= index)
      MEMO.append(Min.of(factor.multiply(Power.of(DoubleScalar.of(1.0 / (index + 1)), exponent)), //
          RealScalar.ONE));
    return MEMO.Get(index);
  }

  /** @return */
  final int maxCount() { // function is not used yet...
    return MEMO.length();
  }
}
