// code by jph
package ch.alpine.subare.ch02.bandits2;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.core.td.SarsaType;
import ch.alpine.subare.core.util.DefaultLearningRate;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.subare.core.util.DiscreteUtils;
import ch.alpine.subare.core.util.DiscreteVs;
import ch.alpine.subare.core.util.LearningRate;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.sca.Chop;
import ch.alpine.tensor.sca.Sign;

class Sarsa_BanditsTest {
  @Test
  public void testSimple() {
    BanditsModel banditsModel = new BanditsModel(10);
    BanditsTrain sarsa_Bandits = new BanditsTrain(banditsModel);
    LearningRate learningRate = DefaultLearningRate.of(RealScalar.of(16), RealScalar.of(1.15));
    DiscreteQsa qsa = sarsa_Bandits.train(SarsaType.ORIGINAL, 100, learningRate);
    DiscreteVs rvs = DiscreteUtils.createVs(banditsModel, sarsa_Bandits.ref);
    DiscreteVs cvs = DiscreteUtils.createVs(banditsModel, qsa);
    Sign.requirePositive(rvs.value(RealScalar.ZERO));
    Chop.NONE.requireAllZero(cvs.value(RealScalar.ONE));
  }
}
