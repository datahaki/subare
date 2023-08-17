// code by jph
package ch.alpine.subare.alg;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.SimpleTestModel;
import ch.alpine.subare.util.SimpleTestModels;
import ch.alpine.tensor.RealScalar;

class ActionValueIterationsTest {
  @Test
  void testSimple() {
    DiscreteQsa qsa = ActionValueIterations.solve(SimpleTestModel.INSTANCE, RealScalar.of(.001));
    SimpleTestModels._checkExact(qsa);
  }
}
