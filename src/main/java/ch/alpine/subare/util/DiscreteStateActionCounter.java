// code by fluric
package ch.alpine.subare.util;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

import ch.alpine.subare.api.DiscreteModel;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.api.StepRecord;
import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.api.ScalarUnaryOperator;
import ch.alpine.tensor.sca.exp.Log;

public class DiscreteStateActionCounter implements StateActionCounter, Serializable {
  private static final ScalarUnaryOperator LOGARITHMIC = scalar -> Log.FUNCTION.apply(scalar.add(RealScalar.ONE));
  // ---
  private final Map<Tensor, Integer> stateActionMap = new HashMap<>();
  private final Map<Tensor, Integer> stateMap = new HashMap<>();

  @Override // from StepDigest
  public void digest(StepRecord stepRecord) {
    Tensor key = StateAction.key(stepRecord);
    Tensor state = stepRecord.prevState();
    stateMap.merge(state, 1, Math::addExact);
    stateActionMap.merge(key, 1, Math::addExact);
  }

  @Override // from StateActionCounter
  public int stateActionCount(Tensor key) {
    return stateActionMap.getOrDefault(key, 0);
  }

  @Override // from StateActionCounter
  public int stateCount(Tensor state) {
    return stateMap.getOrDefault(state, 0);
  }

  @Override // from StateActionCounter
  public boolean isEncountered(Tensor key) {
    return stateActionMap.containsKey(key);
  }

  public void setStateCount(Tensor state, int value) {
    stateMap.put(state, value);
  }

  public void setStateActionCount(Tensor key, int value) {
    stateActionMap.put(key, value);
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
