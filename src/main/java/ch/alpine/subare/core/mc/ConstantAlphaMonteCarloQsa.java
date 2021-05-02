// code by jph
package ch.alpine.subare.core.mc;

import java.util.List;

import ch.alpine.subare.core.DiscountFunction;
import ch.alpine.subare.core.DiscreteModel;
import ch.alpine.subare.core.EpisodeQsaEstimator;
import ch.alpine.subare.core.StateActionCounter;
import ch.alpine.subare.core.StepInterface;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.subare.core.util.LearningRate;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

/** simple every-visit Monte Carlo method suitable for non-stationary environments
 * 
 * (6.1) p.127 */
public class ConstantAlphaMonteCarloQsa extends ConstantAlphaMonteCarloBase implements EpisodeQsaEstimator {
  private final DiscreteQsa qsa;

  /** @param discreteModel
   * @param learningRate
   * @param sac */
  public ConstantAlphaMonteCarloQsa(DiscreteModel discreteModel, LearningRate learningRate, StateActionCounter sac) {
    super(DiscountFunction.of(discreteModel.gamma()), learningRate, sac);
    qsa = DiscreteQsa.build(discreteModel); // <- "arbitrary"
  }

  @Override
  protected void digest(Tensor rewards, List<StepInterface> trajectory) {
    int fromIndex = 0;
    for (StepInterface stepInterface : trajectory) {
      Tensor state = stepInterface.prevState();
      Tensor action = stepInterface.action();
      Scalar gain = discountFunction.apply(rewards.extract(fromIndex, rewards.length()));
      Scalar value0 = qsa.value(state, action);
      Scalar alpha = learningRate.alpha(stepInterface, sac);
      Scalar delta = gain.subtract(value0).multiply(alpha);
      qsa.assign(state, action, value0.add(delta)); // (6.1)
      sac.digest(stepInterface);
      ++fromIndex;
    }
  }

  @Override // from DiscreteQsaSupplier
  public DiscreteQsa qsa() {
    return qsa;
  }
}
