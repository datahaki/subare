// code by jph
package ch.alpine.subare.util;

import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class ExactFeatureMapperTest {
  @Test
  void testFail() {
    assertThrows(Exception.class, () -> ExactFeatureMapper.of(null));
  }
}
