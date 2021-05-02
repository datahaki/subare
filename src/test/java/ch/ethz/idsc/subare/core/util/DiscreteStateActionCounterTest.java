// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensors;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.adapter.StepAdapter;
import junit.framework.TestCase;

public class DiscreteStateActionCounterTest extends TestCase {
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
