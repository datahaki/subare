// code by jph
package ch.alpine.subare.ch02.bandits2;

import ch.alpine.subare.core.td.SarsaType;

/** Double Sarsa for maximization bias */
/* package */ enum Double_Bandits {
  ;
  public static void main(String[] args) {
    BanditsModel banditsModel = new BanditsModel(20);
    BanditsTrain sarsa_Bandits = new BanditsTrain(banditsModel);
    sarsa_Bandits.handle(SarsaType.QLEARNING, 1);
    sarsa_Bandits.handle(SarsaType.EXPECTED, 3);
    sarsa_Bandits.handle(SarsaType.QLEARNING, 2);
  }
}
