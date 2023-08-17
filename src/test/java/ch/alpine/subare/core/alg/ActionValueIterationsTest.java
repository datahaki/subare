// code by jph
package ch.alpine.subare.core.alg;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.subare.core.util.SimpleTestModel;
import ch.alpine.subare.core.util.SimpleTestModels;
import ch.alpine.tensor.RealScalar;

class ActionValueIterationsTest {
  @Test
  void testSimple() {
    DiscreteQsa qsa = ActionValueIterations.solve(SimpleTestModel.INSTANCE, RealScalar.of(.001));
    SimpleTestModels._checkExact(qsa);
  }
}
