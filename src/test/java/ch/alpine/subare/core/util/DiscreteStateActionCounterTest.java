// code by jph
package ch.alpine.subare.core.util;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.core.StateActionCounter;
import ch.alpine.subare.core.StepInterface;
import ch.alpine.subare.core.adapter.StepAdapter;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensors;

public class DiscreteStateActionCounterTest {
  @Test
  public void testSimple() {
    StateActionCounter stateActionCounter = new DiscreteStateActionCounter();
    assertFalse(stateActionCounter.isEncountered(Tensors.vector(1, 2, 3)));
    Scalar scalar = stateActionCounter.stateCount(Tensors.vector(1, 2, 3));
    assertTrue(Scalars.isZero(scalar));
    {
      StepInterface stepInterface = new StepAdapter(Tensors.vector(1, 2, 3), Tensors.vector(3), RealScalar.ZERO, Tensors.vector(1, 5));
      stateActionCounter.digest(stepInterface);
      assertEquals(stateActionCounter.stateCount(Tensors.vector(1, 2, 3)), RealScalar.ONE);
      assertEquals(stateActionCounter.stateActionCount(StateAction.key(stepInterface)), RealScalar.ONE);
    }
    {
      StepInterface stepInterface = new StepAdapter(Tensors.vector(1, 2, 3), Tensors.vector(4), RealScalar.ZERO, Tensors.vector(1, 5));
      assertEquals(stateActionCounter.stateCount(Tensors.vector(1, 2, 3)), RealScalar.ONE);
      assertEquals(stateActionCounter.stateActionCount(StateAction.key(stepInterface)), RealScalar.ZERO);
    }
  }
}
