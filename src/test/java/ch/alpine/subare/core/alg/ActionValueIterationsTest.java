// code by jph
package ch.alpine.subare.core.alg;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.core.adapter.SimpleTestModel;
import ch.alpine.subare.core.adapter.SimpleTestModels;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.tensor.RealScalar;

public class ActionValueIterationsTest {
  @Test
  public void testSimple() {
    DiscreteQsa qsa = ActionValueIterations.solve(SimpleTestModel.INSTANCE, RealScalar.of(.001));
    SimpleTestModels._checkExact(qsa);
  }
}
