// code by jph, fluric
package ch.alpine.subare.util;

import ch.alpine.subare.api.DiscreteModel;
import ch.alpine.subare.api.ExplorationRate;
import ch.alpine.subare.api.QsaInterface;
import ch.alpine.subare.api.StandardModel;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.api.VsInterface;
import ch.alpine.subare.math.FairArg;
import ch.alpine.subare.math.Index;
import ch.alpine.tensor.RationalScalar;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.d.CategoricalDistribution;
import ch.alpine.tensor.pdf.d.DiscreteUniformDistribution;

/** p.33 */
public class EGreedyPolicy extends PolicyBase {
  // TODO SUBARE make explorationRate final
  private ExplorationRate explorationRate;

  public EGreedyPolicy(DiscreteModel discreteModel, QsaInterface qsa, StateActionCounter sac) {
    super(discreteModel, qsa, sac);
    explorationRate = ConstantExplorationRate.of(0.1); // TODO SUBARE magic const
  }

  public EGreedyPolicy(StandardModel standardModel, VsInterface vs, StateActionCounter sac) {
    super(standardModel, vs, sac);
    explorationRate = ConstantExplorationRate.of(0.1); // TODO SUBARE magic const
  }

  public void setExplorationRate(ExplorationRate explorationRate) {
    this.explorationRate = explorationRate;
  }

  @Override
  public Tensor getBestActions(Tensor state) {
    Tensor actions = discreteModel.actions(state);
    Tensor va = Tensor.of(actions.stream().map(action -> qsa.value(state, action)));
    FairArg fairArgMax = FairArg.max(va);
    return Tensor.of(fairArgMax.options().stream().map(actions::get));
  }

  @Override
  public Distribution getDistribution(Tensor state) {
    Tensor bestActions = getBestActions(state);
    Index index = Index.build(bestActions);
    final int optimalCount = bestActions.length();
    final int nonOptimalCount = discreteModel.actions(state).length() - optimalCount;
    Scalar epsilon = explorationRate.epsilon(state, sac);
    // TODO SUBARE check logic
    if (nonOptimalCount == 0) {
      return DiscreteUniformDistribution.of(0, bestActions.length());
      // Tensor pdf = Tensors.vector(v -> RationalScalar.of(1, optimalCount), bestActions.length());
      // return EmpiricalDistribution.fromUnscaledPDF(pdf);
    }
    Tensor pdf = Tensor.of(discreteModel.actions(state).stream() //
        .map(action -> index.containsKey(action) //
            ? RealScalar.ONE.subtract(epsilon).divide(RealScalar.of(optimalCount))
            : epsilon.divide(RealScalar.of(nonOptimalCount))));
    return CategoricalDistribution.fromUnscaledPDF(pdf);
  }

  @Override
  public Scalar probability(Tensor state, Tensor action) {
    Tensor bestActions = getBestActions(state);
    Index index = Index.build(bestActions);
    final int optimalCount = bestActions.length();
    final int nonOptimalCount = discreteModel.actions(state).length() - optimalCount;
    if (nonOptimalCount == 0) // no non-optimal action exists
      return RationalScalar.of(1, optimalCount);
    Scalar epsilon = explorationRate.epsilon(state, sac);
    if (index.containsKey(action))
      return RealScalar.ONE.subtract(epsilon).divide(RealScalar.of(optimalCount));
    return epsilon.divide(RealScalar.of(nonOptimalCount));
  }

  @Override // from PolicyBase
  public EGreedyPolicy copy() {
    EGreedyPolicy eGreedyPolicy = new EGreedyPolicy(discreteModel, qsa, sac);
    eGreedyPolicy.setExplorationRate(explorationRate);
    return eGreedyPolicy;
  }
}
