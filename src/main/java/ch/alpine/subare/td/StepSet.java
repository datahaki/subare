// code by jph
package ch.alpine.subare.td;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import ch.alpine.subare.api.StepRecord;
import ch.alpine.subare.util.StateAction;
import ch.alpine.tensor.Tensor;

/* package */ class StepSet {
  private final Map<Tensor, StepRecord> map = new HashMap<>();

  void register(StepRecord stepRecord) {
    map.putIfAbsent(StateAction.key(stepRecord), stepRecord);
  }

  Collection<StepRecord> values() {
    return map.values();
  }
}
