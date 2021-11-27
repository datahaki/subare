// code by jph and fluric
package ch.alpine.subare.core.td;

import ch.alpine.subare.core.DiscreteModel;
import ch.alpine.subare.core.util.PolicyExt;
import ch.alpine.subare.core.util.PolicyWrap;
import ch.alpine.subare.core.util.StateAction;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

/* package */ class AbstractSarsaEvaluation implements SarsaEvaluation {
  final DiscreteModel discreteModel;

  public AbstractSarsaEvaluation(DiscreteModel discreteModel) {
    this.discreteModel = discreteModel;
  }

  @Override
  public final Scalar evaluate(Tensor state, PolicyExt policy) {
    return discreteModel.actions(state).stream() //
        .anyMatch(action -> policy.sac().isEncountered(StateAction.key(state, action))) //
            ? crossEvaluate(state, policy, policy)
            : RealScalar.ZERO;
  }

  @Override
  public Scalar crossEvaluate(Tensor state, PolicyExt policy1, PolicyExt policy2) {
    Tensor action = new PolicyWrap(policy1).next(state, discreteModel);
    return policy2.qsaInterface().value(state, action);
  }
}