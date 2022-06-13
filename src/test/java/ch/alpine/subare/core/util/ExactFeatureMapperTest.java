// code by jph
package ch.alpine.subare.core.util;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.ch04.gambler.GamblerModel;
import ch.alpine.subare.core.MonteCarloInterface;

class ExactFeatureMapperTest {
  @Test
  void testSimple() {
    MonteCarloInterface monteCarloInterface = GamblerModel.createDefault();
    FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
    assertEquals(featureMapper.featureSize(), 2500);
  }

  @Test
  void testFail() {
    assertThrows(Exception.class, () -> ExactFeatureMapper.of(null));
  }
}
