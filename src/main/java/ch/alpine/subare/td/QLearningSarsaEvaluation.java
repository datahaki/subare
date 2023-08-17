// code by jph, fluric
package ch.alpine.subare.td;

import java.util.Objects;

import ch.alpine.subare.api.DiscreteModel;
import ch.alpine.subare.api.PolicyExt;
import ch.alpine.subare.math.FairArg;
import ch.alpine.subare.util.StateAction;
import ch.alpine.tensor.RationalScalar;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.red.Max;

/* package */ class QLearningSarsaEvaluation implements SarsaEvaluation {
  private final DiscreteModel discreteModel;

  public QLearningSarsaEvaluation(DiscreteModel discreteModel) {
    this.discreteModel = Objects.requireNonNull(discreteModel);
  }

  @Override // from SarsaEvaluation
  public Scalar evaluate(Tensor state, PolicyExt policy) {
    return discreteModel.actions(state).stream() //
        .filter(action -> policy.sac().isEncountered(StateAction.key(state, action))) //
        .map(action -> policy.qsaInterface().value(state, action)) //
        .reduce(Max::of) //
        .orElse(RealScalar.ZERO);
  }

  @Override // from SarsaEvaluation
  public Scalar crossEvaluate(Tensor state, PolicyExt policy1, PolicyExt policy2) {
    Scalar value = RealScalar.ZERO;
    Tensor actions = Tensor.of(discreteModel.actions(state).stream() //
        .filter(action -> policy1.sac().isEncountered(StateAction.key(state, action))));
    if (Tensors.isEmpty(actions))
      return RealScalar.ZERO;
    Tensor eval = Tensor.of(actions.stream().map(action -> policy1.qsaInterface().value(state, action)));
    FairArg fairArgMax = FairArg.max(eval);
    Scalar weight = RationalScalar.of(1, fairArgMax.optionsCount()); // uniform distribution among best actions
    for (int index : fairArgMax.options()) {
      Tensor action = actions.get(index);
      value = value.add(policy2.qsaInterface().value(state, action).multiply(weight)); // use Qsa2 to evaluate state-action pair
    }
    return value;
  }
}
