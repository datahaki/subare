// code by jph
package ch.alpine.subare.ch04.gambler;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.core.td.SarsaType;
import ch.alpine.subare.core.util.DefaultLearningRate;
import ch.alpine.subare.core.util.LearningRate;
import ch.alpine.tensor.RationalScalar;
import ch.alpine.tensor.RealScalar;

class Sarsa_GamblerTest {
  @Test
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
