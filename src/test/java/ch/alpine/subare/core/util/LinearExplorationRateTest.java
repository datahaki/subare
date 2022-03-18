// code by jph
package ch.alpine.subare.core.util;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.util.AssertFail;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.mat.Tolerance;

public class LinearExplorationRateTest {
  @Test
  public void testSimple() {
    LinearExplorationRate.of(10, 1, .5);
    LinearExplorationRate.of(10, 1, 1);
    AssertFail.of(() -> LinearExplorationRate.of(10, .1, .2));
  }

  @Test
  public void testValue() {
    LinearExplorationRate explorationRate = (LinearExplorationRate) LinearExplorationRate.of(10, 0.7, .2);
    Tolerance.CHOP.requireClose(explorationRate.epsilon(RealScalar.of(0)), RealScalar.of(0.7));
    Tolerance.CHOP.requireClose(explorationRate.epsilon(RealScalar.of(5)), RealScalar.of(0.45));
    Tolerance.CHOP.requireClose(explorationRate.epsilon(RealScalar.of(10)), RealScalar.of(0.2));
  }
}
