// code by jph
package ch.alpine.subare.util;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.api.StepRecord;
import ch.alpine.subare.pol.StateActionCounter;
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
      StepRecord stepRecord = new StepRecord(Tensors.vector(1, 2, 3), Tensors.vector(3), RealScalar.ZERO, Tensors.vector(1, 5));
      stateActionCounter.digest(stepRecord);
      assertEquals(stateActionCounter.stateCount(Tensors.vector(1, 2, 3)), 1);
      assertEquals(stateActionCounter.stateActionCount(StateAction.key(stepRecord)), 1);
    }
    {
      StepRecord stepRecord = new StepRecord(Tensors.vector(1, 2, 3), Tensors.vector(4), RealScalar.ZERO, Tensors.vector(1, 5));
      assertEquals(stateActionCounter.stateCount(Tensors.vector(1, 2, 3)), 1);
      assertEquals(stateActionCounter.stateActionCount(StateAction.key(stepRecord)), 0);
    }
  }
}
