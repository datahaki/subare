// code by jph
package ch.alpine.subare.mc;

import java.util.List;

import ch.alpine.subare.api.DiscountFunction;
import ch.alpine.subare.api.DiscreteModel;
import ch.alpine.subare.api.EpisodeVsEstimator;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.api.StepRecord;
import ch.alpine.subare.api.VsInterface;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.DiscreteVs;
import ch.alpine.subare.util.LearningRate;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

/** simple every-visit Monte Carlo method suitable for non-stationary environments
 * 
 * (6.1) p.127 */
public class ConstantAlphaMonteCarloVs extends ConstantAlphaMonteCarloBase implements EpisodeVsEstimator {
  public static EpisodeVsEstimator create(DiscreteModel discreteModel, LearningRate learningRate) {
    return new ConstantAlphaMonteCarloVs( //
        DiscountFunction.of(discreteModel.gamma()), //
        DiscreteVs.build(discreteModel.states()), //
        learningRate, new DiscreteStateActionCounter());
  }

  // ---
  private final VsInterface vs;

  private ConstantAlphaMonteCarloVs(DiscountFunction discountFunction, VsInterface vs, LearningRate learningRate, StateActionCounter sac) {
    super(discountFunction, learningRate, sac);
    this.vs = vs;
  }

  @Override
  protected void digest(Tensor rewards, List<StepRecord> trajectory) {
    int fromIndex = 0;
    for (StepRecord stepInterface : trajectory) {
      Tensor state = stepInterface.prevState();
      Scalar gain = discountFunction.apply(rewards.extract(fromIndex, rewards.length()));
      Scalar value0 = vs.value(state);
      Scalar alpha = learningRate.alpha(stepInterface, sac);
      Scalar delta = gain.subtract(value0).multiply(alpha);
      vs.increment(state, delta); // (6.1)
      sac.digest(stepInterface);
      ++fromIndex;
    }
  }

  @Override // from DiscreteVsSupplier
  public DiscreteVs vs() {
    return (DiscreteVs) vs;
  }
}
