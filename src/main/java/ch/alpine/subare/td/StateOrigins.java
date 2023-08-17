// code by jph
package ch.alpine.subare.td;

import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import ch.alpine.subare.api.StepDigest;
import ch.alpine.subare.api.StepRecord;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Throw;

/* package */ class StateOrigins implements StepDigest {
  private final Map<Tensor, StepSet> map = new HashMap<>();

  /** @param state
   * @return step interfaces with nextState == state */
  Collection<StepRecord> values(Tensor state) {
    if (map.containsKey(state)) {
      // TODO SUBARE this is a preliminary check only during development
      Collection<StepRecord> collection = map.get(state).values();
      for (StepRecord stepInterface : collection)
        if (!stepInterface.nextState().equals(state))
          throw new Throw(state);
      return collection;
    }
    return Collections.emptyList();
  }

  @Override
  public void digest(StepRecord stepInterface) {
    // TODO SUBARE code redundant to StepSet
    map.computeIfAbsent(stepInterface.nextState(), i -> new StepSet()).register(stepInterface);
  }
}
