// code by jph
package ch.alpine.subare.ch06.maxbias;

import ch.alpine.subare.core.StateActionCounter;
import ch.alpine.subare.core.td.Sarsa;
import ch.alpine.subare.core.td.SarsaType;
import ch.alpine.subare.core.util.ActionValueStatistics;
import ch.alpine.subare.core.util.DefaultLearningRate;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.subare.core.util.DiscreteStateActionCounter;
import ch.alpine.subare.core.util.DiscreteUtils;
import ch.alpine.subare.core.util.DiscreteVs;
import ch.alpine.subare.core.util.EGreedyPolicy;
import ch.alpine.subare.core.util.ExploringStarts;
import ch.alpine.subare.core.util.Infoline;
import ch.alpine.subare.core.util.LearningRate;
import ch.alpine.subare.core.util.LinearExplorationRate;
import ch.alpine.subare.core.util.PolicyType;
import ch.alpine.tensor.sca.Round;

enum Sarsa_Maxbias {
  ;
  static void handle(SarsaType sarsaType, int nstep) {
    System.out.println(sarsaType);
    Maxbias maxbias = new Maxbias(3);
    final DiscreteQsa ref = MaxbiasHelper.getOptimalQsa(maxbias);
    int batches = 100;
    DiscreteQsa qsa = DiscreteQsa.build(maxbias);
    LearningRate learningRate = DefaultLearningRate.of(2, 0.6);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(maxbias, qsa, sac);
    policy.setExplorationRate(LinearExplorationRate.of(batches, 0.7, 0.1));
    Sarsa sarsa = sarsaType.sarsa(maxbias, learningRate, qsa, sac, policy);
    ActionValueStatistics avs = new ActionValueStatistics(maxbias);
    for (int index = 0; index < batches; ++index) {
      if (batches - 10 < index)
        Infoline.print(maxbias, index, ref, qsa);
      // sarsa.supplyPolicy(() -> policy);
      ExploringStarts.batch(maxbias, policy, nstep, sarsa, avs);
    }
    DiscreteVs vs = DiscreteUtils.createVs(maxbias, qsa);
    DiscreteUtils.print(vs, Round._3);
  }

  public static void main(String[] args) {
    handle(SarsaType.ORIGINAL, 3);
    handle(SarsaType.EXPECTED, 3);
    handle(SarsaType.QLEARNING, 3);
  }
}
