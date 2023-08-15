// code by jz and jph
package ch.alpine.subare.core.util;

import ch.alpine.subare.core.StateActionCounter;
import ch.alpine.subare.core.StepRecord;
import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.RationalScalar;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.Throw;
import ch.alpine.tensor.red.Min;
import ch.alpine.tensor.sca.Sign;
import ch.alpine.tensor.sca.pow.Power;

/** stochastic approximation theory
 * p.35 equation (2.7)
 * 
 * conditions required for convergence with probability 1:
 * sum_n alpha_n(s, a)^1 == infinity
 * sum_n alpha_n(s, a)^2 < infinity
 * 
 * Example:
 * in the Gambler problem the following values seem to work well
 * OriginalSarsa factor == 1.3, and exponent == 0.51
 * QLearning factor == 0.2, and exponent == 0.55 */
/* package */ abstract class DecayedLearningRate implements LearningRate {
  private final Scalar factor;
  private final Scalar exponent;
  /** lookup table to speed up computation */
  private final Tensor memo = Tensors.vector(1.0); // index == 0 => learning rate == 1

  protected DecayedLearningRate(Scalar factor, Scalar exponent) {
    if (Scalars.lessEquals(exponent, RationalScalar.HALF))
      throw new Throw(factor, exponent);
    this.factor = Sign.requirePositive(factor);
    this.exponent = exponent;
  }

  @Override // from LearningRate
  public final synchronized Scalar alpha(StepRecord stepInterface, StateActionCounter stateActionCounter) {
    Tensor key = StateAction.key(stepInterface);
    int index = Scalars.intValueExact(stateActionCounter.stateActionCount(key));
    while (memo.length() <= index)
      memo.append(Min.of( // TODO SUBARE the "+1" in the denominator may not be ideal... perhaps +0.5, or +0 ?
          factor.multiply(Power.of(DoubleScalar.of(1.0 / (index + 1)), exponent)), //
          RealScalar.ONE));
    return memo.Get(index);
  }

  /** @return */
  final int maxCount() { // function is not used yet...
    return memo.length();
  }
}
