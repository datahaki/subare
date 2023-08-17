// code by jph
package ch.alpine.subare.core.util;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

import ch.alpine.subare.core.api.MoveInterface;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Throw;

/** for deterministic {@link MoveInterface} to precompute and store the effective actions */
public class StateActionMap {
  private final Map<Tensor, Tensor> map;

  public StateActionMap(Map<Tensor, Tensor> map) {
    this.map = map;
  }

  public StateActionMap() {
    this(new HashMap<>());
  }

  /** @param state
   * @return unmodifiable tensor of actions */
  public Tensor actions(Tensor state) {
    return Objects.requireNonNull(map.get(state));
  }

  /** @param state
   * @param actions
   * @throws Exception if state already exists as key in this map */
  public void put(Tensor state, Tensor actions) {
    if (map.containsKey(state))
      throw new Throw(state);
    map.put(state, actions.unmodifiable());
  }
}
