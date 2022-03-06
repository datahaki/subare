// code by jph
package ch.alpine.subare.ch04.gambler;

import java.io.File;

import ch.alpine.subare.core.td.SarsaType;
import ch.alpine.subare.core.util.DefaultLearningRate;
import ch.alpine.subare.core.util.LearningRate;
import ch.alpine.tensor.RationalScalar;
import ch.alpine.tensor.RealScalar;
import junit.framework.TestCase;

public class Sarsa_GamblerTest extends TestCase {
  public void testSimple() throws Exception {
    for (SarsaType sarsaType : SarsaType.values()) {
      GamblerModel gamblerModel = new GamblerModel(20, RationalScalar.of(4, 10));
      Sarsa_Gambler sarsa_Gambler = new Sarsa_Gambler(gamblerModel);
      LearningRate learningRate = DefaultLearningRate.of(RealScalar.of(3), RealScalar.of(0.81));
      sarsa_Gambler.train(sarsaType, 10, learningRate);
      {
        File file = Sarsa_Gambler.getGifFileQsa(sarsaType);
        assertTrue(file.isFile());
        file.delete();
      }
      {
        File file = Sarsa_Gambler.getGifFileSac(sarsaType);
        assertTrue(file.isFile());
        file.delete();
      }
    }
  }
}
