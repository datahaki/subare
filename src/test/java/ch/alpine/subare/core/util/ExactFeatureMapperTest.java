// code by jph
package ch.alpine.subare.core.util;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.ch04.gambler.GamblerModel;
import ch.alpine.subare.core.MonteCarloInterface;
import ch.alpine.subare.util.AssertFail;

public class ExactFeatureMapperTest {
  @Test
  public void testSimple() {
    MonteCarloInterface monteCarloInterface = GamblerModel.createDefault();
    FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
    assertEquals(featureMapper.featureSize(), 2500);
  }

  @Test
  public void testFail() {
    AssertFail.of(() -> ExactFeatureMapper.of(null));
  }
}
