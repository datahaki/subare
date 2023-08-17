// code by jph, fluric
package ch.alpine.subare.td;

import ch.alpine.subare.api.DiscreteModel;
import ch.alpine.subare.api.PolicyExt;
import ch.alpine.subare.util.StateAction;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;

/* package */ class ExpectedSarsaEvaluation extends AbstractSarsaEvaluation {
  public ExpectedSarsaEvaluation(DiscreteModel discreteModel) {
    super(discreteModel);
  }

  @Override
  public Scalar crossEvaluate(Tensor state, PolicyExt policy1, PolicyExt policy2) {
    Tensor actions = Tensor.of(discreteModel.actions(state).stream() //
        .filter(action -> policy1.sac().isEncountered(StateAction.key(state, action))));
    if (Tensors.isEmpty(actions))
      return RealScalar.ZERO;
    return actions.stream() //
        .map(action -> policy1.probability(state, action).multiply(policy2.qsaInterface().value(state, action))) //
        .reduce(Scalar::add) //
        .orElseThrow();
  }
}
