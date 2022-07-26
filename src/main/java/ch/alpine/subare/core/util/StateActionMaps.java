// code by jph
package ch.alpine.subare.core.util;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import ch.alpine.subare.core.MoveInterface;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.Throw;

/** for deterministic {@link MoveInterface} to precompute and store the effective actions */
public enum StateActionMaps {
  ;
  /** collects effective actions of deterministic move interface, i.e. for each
   * state a subset of actions that reaches all states that are possible to reach
   * 
   * @param states
   * @param actions
   * @param moveInterface deterministic
   * @return */
  public static StateActionMap build(Tensor states, Tensor actions, MoveInterface moveInterface) {
    Map<Tensor, Tensor> map = new HashMap<>();
    for (Tensor state : states) {
      Tensor filter = Tensors.empty(); // effective actions
      Set<Tensor> set = new HashSet<>();
      for (Tensor action : actions) {
        Tensor next = moveInterface.move(state, action);
        if (set.add(next))
          filter.append(action);
      }
      if (Tensors.isEmpty(filter))
        throw new Throw(state); // missing actions
      map.put(state, filter.unmodifiable());
    }
    return new StateActionMap(map);
  }
}
