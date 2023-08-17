// code by jph
package ch.alpine.subare.book.ch06.maxbias;

import ch.alpine.subare.api.EpisodeInterface;
import ch.alpine.subare.api.Policy;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.api.StepRecord;
import ch.alpine.subare.td.DoubleSarsa;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.util.DefaultLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.DiscreteVs;
import ch.alpine.subare.util.EGreedyPolicy;
import ch.alpine.subare.util.EpisodeKickoff;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.LinearExplorationRate;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.Put;

/** Double Sarsa for maximization bias */
enum Double_Maxbias {
  ;
  static void handle(SarsaType sarsaType, int n) throws Exception {
    System.out.println("double " + sarsaType);
    Maxbias maxbias = new Maxbias(10);
    final DiscreteQsa ref = MaxbiasHelper.getOptimalQsa(maxbias);
    int batches = 10;
    DiscreteQsa qsa1 = DiscreteQsa.build(maxbias);
    DiscreteQsa qsa2 = DiscreteQsa.build(maxbias);
    StateActionCounter sac1 = new DiscreteStateActionCounter();
    EGreedyPolicy policy1 = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(maxbias, qsa1, sac1);
    policy1.setExplorationRate(LinearExplorationRate.of(batches, 0.1, 0.01));
    StateActionCounter sac2 = new DiscreteStateActionCounter();
    EGreedyPolicy policy2 = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(maxbias, qsa2, sac2);
    policy1.setExplorationRate(LinearExplorationRate.of(batches, 0.1, 0.01));
    DoubleSarsa doubleSarsa = sarsaType.doubleSarsa( //
        maxbias, DefaultLearningRate.of(5, .51), //
        qsa1, qsa2, sac1, sac2, policy1, policy2);
    for (int index = 0; index < batches; ++index) {
      if (batches - 10 < index)
        Infoline.print(maxbias, index, ref, qsa1);
      ExploringStarts.batch(maxbias, doubleSarsa.getPolicy(), n, doubleSarsa);
    }
    // qsa.print(Round.toMultipleOf(DecimalScalar.of(.01)));
    System.out.println("---");
    DiscreteVs vs = DiscreteUtils.createVs(maxbias, qsa1);
    Put.of(HomeDirectory.file("gridworld_" + sarsaType), vs.values());
    Policy policy = PolicyType.GREEDY.bestEquiprobable(maxbias, doubleSarsa.qsa(), doubleSarsa.sac());
    EpisodeInterface ei = EpisodeKickoff.single(maxbias, policy);
    while (ei.hasNext()) {
      StepRecord stepInterface = ei.step();
      Tensor state = stepInterface.prevState();
      System.out.println(state + " then " + stepInterface.action());
    }
  }

  public static void main(String[] args) throws Exception {
    handle(SarsaType.ORIGINAL, 1);
    handle(SarsaType.EXPECTED, 1);
    handle(SarsaType.QLEARNING, 1);
  }
}
