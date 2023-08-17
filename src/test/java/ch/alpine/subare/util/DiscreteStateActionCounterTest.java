// code by jph
package ch.alpine.subare.util;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.api.StepRecord;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensors;

class DiscreteStateActionCounterTest {
  @Test
  void testSimple() {
    StateActionCounter stateActionCounter = new DiscreteStateActionCounter();
    assertFalse(stateActionCounter.isEncountered(Tensors.vector(1, 2, 3)));
    int scalar = stateActionCounter.stateCount(Tensors.vector(1, 2, 3));
    assertEquals(scalar, 0);
    {
      StepRecord stepInterface = new StepRecord(Tensors.vector(1, 2, 3), Tensors.vector(3), RealScalar.ZERO, Tensors.vector(1, 5));
      stateActionCounter.digest(stepInterface);
      assertEquals(stateActionCounter.stateCount(Tensors.vector(1, 2, 3)), 1);
      assertEquals(stateActionCounter.stateActionCount(StateAction.key(stepInterface)), 1);
    }
    {
      StepRecord stepInterface = new StepRecord(Tensors.vector(1, 2, 3), Tensors.vector(4), RealScalar.ZERO, Tensors.vector(1, 5));
      assertEquals(stateActionCounter.stateCount(Tensors.vector(1, 2, 3)), 1);
      assertEquals(stateActionCounter.stateActionCount(StateAction.key(stepInterface)), 0);
    }
  }
}
