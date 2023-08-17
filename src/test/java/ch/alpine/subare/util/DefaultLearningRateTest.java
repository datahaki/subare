// code by jph
package ch.alpine.subare.util;

import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class DefaultLearningRateTest {
  @Test
  void testFailFactor() {
    assertThrows(Exception.class, () -> DefaultLearningRate.of(0, 1));
    assertThrows(Exception.class, () -> DefaultLearningRate.of(-1, 1));
  }

  @Test
  void testFailExponent() {
    assertThrows(Exception.class, () -> DefaultLearningRate.of(1, 0.5));
    assertThrows(Exception.class, () -> DefaultLearningRate.of(1, 0.4));
  }
}
