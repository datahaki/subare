// code by jph
package ch.alpine.subare.core.td;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import ch.alpine.subare.core.StepRecord;
import ch.alpine.subare.core.util.StateAction;
import ch.alpine.tensor.Tensor;

/* package */ class StepSet {
  private final Map<Tensor, StepRecord> map = new HashMap<>();

  void register(StepRecord stepInterface) {
    map.putIfAbsent(StateAction.key(stepInterface), stepInterface);
  }

  Collection<StepRecord> values() {
    return map.values();
  }
}
