// code by jph
package ch.alpine.subare.util;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;

class StateActionMapTest {
  @Test
  void testSimple() {
    StateActionMap stateActionMap = new StateActionMap();
    Tensor key = Tensors.vector(1);
    Tensor values = Tensors.vector(1, 2);
    stateActionMap.put(key, values);
    Tensor actions = stateActionMap.actions(Tensors.vector(1));
    assertEquals(actions, values);
  }

  @Test
  void testDuplicateFail() {
    StateActionMap stateActionMap = new StateActionMap();
    stateActionMap.put(Tensors.vector(1), Tensors.vector(1, 2));
    assertThrows(Exception.class, () -> stateActionMap.put(Tensors.vector(1), Tensors.vector(1, 2)));
  }
}
