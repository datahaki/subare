// code by jph
package ch.ethz.idsc.subare.core.alg;

import ch.alpine.tensor.RealScalar;
import ch.ethz.idsc.subare.core.adapter.SimpleTestModel;
import ch.ethz.idsc.subare.core.adapter.SimpleTestModels;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import junit.framework.TestCase;

public class ActionValueIterationsTest extends TestCase {
  public void testSimple() {
    DiscreteQsa qsa = ActionValueIterations.solve(SimpleTestModel.INSTANCE, RealScalar.of(.001));
    SimpleTestModels._checkExact(qsa);
  }
}
