// code by jph
package ch.ethz.idsc.subare.core.td;

import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.TensorRuntimeException;
import ch.ethz.idsc.subare.core.StepDigest;
import ch.ethz.idsc.subare.core.StepInterface;

/* package */ class StateOrigins implements StepDigest {
  private final Map<Tensor, StepSet> map = new HashMap<>();

  /** @param state
   * @return step interfaces with nextState == state */
  Collection<StepInterface> values(Tensor state) {
    if (map.containsKey(state)) {
      // TODO this is a preliminary check only during development
      Collection<StepInterface> collection = map.get(state).values();
      for (StepInterface stepInterface : collection)
        if (!stepInterface.nextState().equals(state))
          throw TensorRuntimeException.of(state);
      return collection;
    }
    return Collections.emptyList();
  }

  @Override
  public void digest(StepInterface stepInterface) {
    // TODO code redudant to StepSet
    Tensor key = stepInterface.nextState();
    if (!map.containsKey(key))
      map.put(key, new StepSet());
    map.get(key).register(stepInterface);
  }
}
