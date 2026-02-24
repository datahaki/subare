// code by jph
package ch.alpine.subare.alg;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.SimpleTestModel;
import ch.alpine.subare.util.SimpleTestModels;
import ch.alpine.tensor.sca.Chop;

class ActionValueIterationTest {
  @Test
  void testSimple() {
    DiscreteQsa qsa = ActionValueIteration.solve(SimpleTestModel.INSTANCE, Chop._03);
    SimpleTestModels._checkExact(qsa);
  }
}
