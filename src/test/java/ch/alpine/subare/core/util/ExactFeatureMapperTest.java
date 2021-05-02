// code by jph
package ch.alpine.subare.core.util;

import ch.alpine.subare.ch04.gambler.GamblerModel;
import ch.alpine.subare.core.MonteCarloInterface;
import ch.alpine.subare.util.AssertFail;
import junit.framework.TestCase;

public class ExactFeatureMapperTest extends TestCase {
  public void testSimple() {
    MonteCarloInterface monteCarloInterface = GamblerModel.createDefault();
    FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
    assertEquals(featureMapper.featureSize(), 2500);
  }

  public void testFail() {
    AssertFail.of(() -> ExactFeatureMapper.of(null));
  }
}
