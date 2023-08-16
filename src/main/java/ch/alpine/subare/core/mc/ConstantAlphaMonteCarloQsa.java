// code by jph
package ch.alpine.subare.core.mc;

import java.util.List;

import ch.alpine.subare.core.DiscountFunction;
import ch.alpine.subare.core.DiscreteModel;
import ch.alpine.subare.core.EpisodeQsaEstimator;
import ch.alpine.subare.core.StateActionCounter;
import ch.alpine.subare.core.StepRecord;
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
  protected void digest(Tensor rewards, List<StepRecord> trajectory) {
    int fromIndex = 0;
    for (StepRecord stepRecord : trajectory) {
      Tensor state = stepRecord.prevState();
      Tensor action = stepRecord.action();
      Scalar gain = discountFunction.apply(rewards.extract(fromIndex, rewards.length()));
      Scalar alpha = learningRate.alpha(stepRecord, sac);
      qsa.blend(state, action, gain, alpha); // (6.1)
      sac.digest(stepRecord);
      ++fromIndex;
    }
  }

  @Override // from DiscreteQsaSupplier
  public DiscreteQsa qsa() {
    return qsa;
  }
}
