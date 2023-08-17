// code by jph
package ch.alpine.subare.util;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.api.StepRecord;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensors;

class DiscreteStateActionCounterTest {
  @Test
  void testSimple() {
    StateActionCounter stateActionCounter = new DiscreteStateActionCounter();
    assertFalse(stateActionCounter.isEncountered(Tensors.vector(1, 2, 3)));
    Scalar scalar = stateActionCounter.stateCount(Tensors.vector(1, 2, 3));
    assertTrue(Scalars.isZero(scalar));
    {
      StepRecord stepInterface = new StepRecord(Tensors.vector(1, 2, 3), Tensors.vector(3), RealScalar.ZERO, Tensors.vector(1, 5));
      stateActionCounter.digest(stepInterface);
      assertEquals(stateActionCounter.stateCount(Tensors.vector(1, 2, 3)), RealScalar.ONE);
      assertEquals(stateActionCounter.stateActionCount(StateAction.key(stepInterface)), RealScalar.ONE);
    }
    {
      StepRecord stepInterface = new StepRecord(Tensors.vector(1, 2, 3), Tensors.vector(4), RealScalar.ZERO, Tensors.vector(1, 5));
      assertEquals(stateActionCounter.stateCount(Tensors.vector(1, 2, 3)), RealScalar.ONE);
      assertEquals(stateActionCounter.stateActionCount(StateAction.key(stepInterface)), RealScalar.ZERO);
    }
  }
}
