// code by jph
package ch.alpine.subare.book.ch02.bandits2;

import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.td.DoubleSarsa;
import ch.alpine.subare.td.Sarsa;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.util.DefaultLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.EGreedyPolicy;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.LearningRate;
import ch.alpine.subare.util.LinearExplorationRate;
import ch.alpine.subare.util.Loss;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Subdivide;
import ch.alpine.tensor.sca.Round;

/** Sarsa applied to bandits */
/* package */ class BanditsTrain {
  private final BanditsModel banditsModel;
  final DiscreteQsa ref;

  public BanditsTrain(BanditsModel banditsModel) {
    this.banditsModel = banditsModel;
    ref = BanditsHelper.getOptimalQsa(banditsModel); // true q-function, for error measurement
  }

  /** @param sarsaType
   * @param batches
   * @param learningRate
   * @return
   * @throws Exception */
  DiscreteQsa train(SarsaType sarsaType, int batches, LearningRate learningRate) {
    Tensor epsilon = Subdivide.of(0.6, 0.01, batches);
    epsilon.unmodifiable();
    DiscreteQsa qsa = DiscreteQsa.build(banditsModel); // q-function for training, initialized to 0
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(banditsModel, qsa, sac);
    policy.setExplorationRate(LinearExplorationRate.of(batches, 0.6, 0.01));
    // ---
    final Sarsa sarsa = sarsaType.sarsa(banditsModel, learningRate, qsa, sac, policy);
    // ---
    for (int index = 0; index < batches; ++index) {
      Scalar error1 = Loss.accumulation(banditsModel, ref, qsa);
      error1.zero();
      // System.out.println(index + " " + epsilon.Get(index).map(Round._2) + " " + error1.map(Round._3));
      ExploringStarts.batch(banditsModel, policy, 1, sarsa);
    }
    return qsa;
  }

  DiscreteQsa handle(SarsaType sarsaType, int n) {
    System.out.println("double " + sarsaType);
    int batches = 100;
    Tensor epsilon = Subdivide.of(0.3, 0.01, batches); // used in egreedy
    DiscreteQsa qsa1 = DiscreteQsa.build(banditsModel);
    DiscreteQsa qsa2 = DiscreteQsa.build(banditsModel);
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(banditsModel, DiscreteQsa.build(banditsModel), new DiscreteStateActionCounter());
    StateActionCounter sac1 = new DiscreteStateActionCounter();
    EGreedyPolicy policy1 = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(banditsModel, qsa1, sac1);
    policy1.setExplorationRate(LinearExplorationRate.of(batches, 0.3, 0.01));
    StateActionCounter sac2 = new DiscreteStateActionCounter();
    EGreedyPolicy policy2 = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(banditsModel, qsa2, sac2);
    policy2.setExplorationRate(LinearExplorationRate.of(batches, 0.3, 0.01));
    DoubleSarsa doubleSarsa = sarsaType.doubleSarsa( //
        banditsModel, //
        DefaultLearningRate.of(15, 1.31), //
        qsa1, qsa2, sac1, sac2, policy1, policy2);
    for (int index = 0; index < batches; ++index) {
      Scalar error = Loss.accumulation(banditsModel, ref, qsa1);
      if (batches - 10 < index)
        System.out.println(index + " " + epsilon.Get(index).map(Round._2) + " " + error.map(Round._3));
      policy.setQsa(doubleSarsa.qsa());
      policy.setSac(doubleSarsa.sac());
      ExploringStarts.batch(banditsModel, policy, n, doubleSarsa);
    }
    System.out.println("---");
    System.out.println("true state values:");
    DiscreteUtils.print(DiscreteUtils.createVs(banditsModel, ref), Round._3);
    System.out.println("estimated state values:");
    DiscreteUtils.print(DiscreteUtils.createVs(banditsModel, qsa1), Round._3);
    return qsa1;
  }
}
