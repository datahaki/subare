// code by jph, fluric
package ch.alpine.subare.td;

import ch.alpine.subare.mod.DiscreteModel;
import ch.alpine.subare.pol.PolicyExt;
import ch.alpine.subare.pol.PolicyWrap;
import ch.alpine.subare.util.StateAction;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;

class OriginalSarsaEvaluation extends AbstractSarsaEvaluation {
  public OriginalSarsaEvaluation(DiscreteModel discreteModel) {
    super(discreteModel);
  }

  @Override
  public Scalar crossEvaluate(Tensor state, PolicyExt policy1, PolicyExt policy2) {
    Tensor actions = Tensor.of(discreteModel.actions(state).stream() //
        .filter(action -> policy1.sac().isEncountered(StateAction.key(state, action))));
    if (Tensors.isEmpty(actions))
      return RealScalar.ZERO;
    Tensor action = new PolicyWrap(policy1).next(state, discreteModel);
    return policy2.qsaInterface().value(state, action);
  }
}
