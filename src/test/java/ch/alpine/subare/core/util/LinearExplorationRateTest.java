// code by jph
package ch.alpine.subare.core.util;

import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.mat.Tolerance;

class LinearExplorationRateTest {
  @Test
  void testSimple() {
    LinearExplorationRate.of(10, 1, .5);
    LinearExplorationRate.of(10, 1, 1);
    assertThrows(Exception.class, () -> LinearExplorationRate.of(10, .1, .2));
  }

  @Test
  void testValue() {
    LinearExplorationRate explorationRate = (LinearExplorationRate) LinearExplorationRate.of(10, 0.7, .2);
    Tolerance.CHOP.requireClose(explorationRate.epsilon(RealScalar.of(0)), RealScalar.of(0.7));
    Tolerance.CHOP.requireClose(explorationRate.epsilon(RealScalar.of(5)), RealScalar.of(0.45));
    Tolerance.CHOP.requireClose(explorationRate.epsilon(RealScalar.of(10)), RealScalar.of(0.2));
  }
}
