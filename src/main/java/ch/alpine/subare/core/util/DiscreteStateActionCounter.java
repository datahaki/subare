// code by fluric
package ch.alpine.subare.core.util;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

import ch.alpine.subare.core.api.DiscreteModel;
import ch.alpine.subare.core.api.StateActionCounter;
import ch.alpine.subare.core.api.StepRecord;
import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.api.ScalarUnaryOperator;
import ch.alpine.tensor.sca.exp.Log;

public class DiscreteStateActionCounter implements StateActionCounter, Serializable {
  private static final ScalarUnaryOperator LOGARITHMIC = scalar -> Log.FUNCTION.apply(scalar.add(RealScalar.ONE));
  // ---
  private final Map<Tensor, Integer> stateActionMap = new HashMap<>();
  private final Map<Tensor, Integer> stateMap = new HashMap<>();

  @Override // from StepDigest
  public void digest(StepRecord stepInterface) {
    Tensor key = StateAction.key(stepInterface);
    Tensor state = stepInterface.prevState();
    stateMap.merge(state, 1, Math::addExact);
    stateActionMap.merge(key, 1, Math::addExact);
  }

  @Override // from StateActionCounter
  public Scalar stateActionCount(Tensor key) {
    return RealScalar.of(stateActionMap.getOrDefault(key, 0));
  }

  @Override // from StateActionCounter
  public Scalar stateCount(Tensor state) {
    return RealScalar.of(stateMap.getOrDefault(state, 0));
  }

  @Override // from StateActionCounter
  public boolean isEncountered(Tensor key) {
    return stateActionMap.containsKey(key);
  }

  public void setStateCount(Tensor state, Scalar value) {
    stateMap.put(state, Scalars.intValueExact(value));
  }

  public void setStateActionCount(Tensor key, Scalar value) {
    stateActionMap.put(key, Scalars.intValueExact(value));
  }

  public Scalar getLogarithmicStateActionCount(Tensor state, Tensor action) {
    Tensor key = StateAction.key(state, action);
    // TODO SUBARE inconsistent LOGARITHMIC[0] == log[0+1] != -Infty
    return stateActionMap.containsKey(key) //
        ? LOGARITHMIC.apply(RealScalar.of(stateActionMap.get(key)))
        : DoubleScalar.NEGATIVE_INFINITY;
  }

  public DiscreteQsa inQsa(DiscreteModel discreteModel) {
    DiscreteQsa qsa = DiscreteQsa.build(discreteModel);
    for (Tensor state : discreteModel.states())
      for (Tensor action : discreteModel.actions(state))
        qsa.assign(state, action, getLogarithmicStateActionCount(state, action));
    return qsa;
  }
}
