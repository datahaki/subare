// code by jph
package ch.alpine.subare.core.alg;

import ch.alpine.subare.core.adapter.SimpleTestModel;
import ch.alpine.subare.core.adapter.SimpleTestModels;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.tensor.RealScalar;
import junit.framework.TestCase;

public class ActionValueIterationsTest extends TestCase {
  public void testSimple() {
    DiscreteQsa qsa = ActionValueIterations.solve(SimpleTestModel.INSTANCE, RealScalar.of(.001));
    SimpleTestModels._checkExact(qsa);
  }
}
