// code by jph
package ch.alpine.subare.book.ch02.bandits2;

import ch.alpine.subare.api.LearningRate;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.util.DefaultLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.DiscreteVs;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.sca.Round;

/* package */ enum Sarsa_Bandits {
  ;
  public static void main(String[] args) {
    BanditsModel banditsModel = new BanditsModel(10);
    BanditsTrain sarsa_Bandits = new BanditsTrain(banditsModel);
    LearningRate learningRate = DefaultLearningRate.of(RealScalar.of(16), RealScalar.of(1.15));
    DiscreteQsa qsa = sarsa_Bandits.train(SarsaType.ORIGINAL, 100, learningRate);
    System.out.println("---");
    System.out.println("true state values:");
    DiscreteVs rvs = DiscreteUtils.createVs(banditsModel, sarsa_Bandits.ref);
    DiscreteUtils.print(rvs, Round._3);
    System.out.println("estimated state values:");
    DiscreteVs cvs = DiscreteUtils.createVs(banditsModel, qsa);
    DiscreteUtils.print(cvs, Round._3);
    System.out.println(rvs.value(RealScalar.ZERO));
  }
}
