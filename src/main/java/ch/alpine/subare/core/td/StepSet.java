// code by jph
package ch.alpine.subare.core.td;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import ch.alpine.subare.core.StepInterface;
import ch.alpine.subare.core.util.StateAction;
import ch.alpine.tensor.Tensor;

/* package */ class StepSet {
  private final Map<Tensor, StepInterface> map = new HashMap<>();

  void register(StepInterface stepInterface) {
    map.computeIfAbsent(StateAction.key(stepInterface), i -> stepInterface);
  }

  Collection<StepInterface> values() {
    return map.values();
  }
}
