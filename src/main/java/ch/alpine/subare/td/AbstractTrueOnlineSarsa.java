// code by jph, fluric
package ch.alpine.subare.td;

import ch.alpine.subare.api.FeatureMapper;
import ch.alpine.subare.api.LearningRate;
import ch.alpine.subare.api.MonteCarloInterface;
import ch.alpine.subare.api.StateActionCounterSupplier;
import ch.alpine.subare.api.TrueOnlineInterface;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.StateAction;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.sca.Clips;

abstract class AbstractTrueOnlineSarsa implements TrueOnlineInterface, StateActionCounterSupplier {
  final MonteCarloInterface monteCarloInterface;
  final FeatureMapper featureMapper;
  final LearningRate learningRate;
  final SarsaEvaluation evaluationType;
  final Scalar gamma;
  final Scalar gamma_lambda;

  protected AbstractTrueOnlineSarsa(MonteCarloInterface monteCarloInterface, SarsaEvaluation evaluationType, //
      Scalar lambda, LearningRate learningRate, //
      FeatureMapper featureMapper) {
    this.monteCarloInterface = monteCarloInterface;
    this.evaluationType = evaluationType;
    this.gamma = monteCarloInterface.gamma();
    gamma_lambda = gamma.multiply(Clips.unit().requireInside(lambda));
    this.learningRate = learningRate;
    this.featureMapper = featureMapper;
  }

  /** Returns the Qsa according to the current feature weights.
   * Only use this function, when the state-action space is small enough. */
  @Override // from DiscreteQsaSupplier
  public final DiscreteQsa qsa() {
    DiscreteQsa qsa = DiscreteQsa.build(monteCarloInterface);
    for (Tensor state : monteCarloInterface.states())
      for (Tensor action : monteCarloInterface.actions(state)) {
        Tensor stateActionPair = StateAction.key(state, action);
        qsa.assign(state, action, (Scalar) featureMapper.getFeature(stateActionPair).dot(getW()));
      }
    return qsa;
  }

  public abstract Tensor getW();
}
