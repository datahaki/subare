// code by jph
package ch.alpine.subare.ch02.bandits2;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.core.td.SarsaType;

class Double_BanditsTest {
  @Test
  void testSimple() {
    BanditsModel banditsModel = new BanditsModel(20);
    BanditsTrain sarsa_Bandits = new BanditsTrain(banditsModel);
    sarsa_Bandits.handle(SarsaType.QLEARNING, 1);
    sarsa_Bandits.handle(SarsaType.EXPECTED, 3);
    sarsa_Bandits.handle(SarsaType.QLEARNING, 2);
  }
}
