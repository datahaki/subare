// code by jph
package ch.alpine.subare.book.ch04.grid;

import java.util.concurrent.TimeUnit;

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
import ch.alpine.subare.util.gfx.StateActionRasters;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;
import ch.alpine.tensor.io.Put;

/** Double Sarsa for gridworld */
enum Double_Gridworld {
  ;
  static void handle(SarsaType sarsaType, int nstep) throws Exception {
    System.out.println("double " + sarsaType);
    Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    int batches = 40;
    DiscreteQsa qsa1 = DiscreteQsa.build(gridworld);
    DiscreteQsa qsa2 = DiscreteQsa.build(gridworld);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gridworld, DiscreteQsa.build(gridworld), sac);
    policy.setExplorationRate(LinearExplorationRate.of(batches, 0.1, 0.01));
    StateActionCounter sac1 = new DiscreteStateActionCounter();
    EGreedyPolicy policy1 = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gridworld, qsa1, sac1);
    policy1.setExplorationRate(LinearExplorationRate.of(batches, 0.1, 0.01));
    StateActionCounter sac2 = new DiscreteStateActionCounter();
    EGreedyPolicy policy2 = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gridworld, qsa2, sac2);
    policy2.setExplorationRate(LinearExplorationRate.of(batches, 0.1, 0.01));
    DoubleSarsa doubleSarsa = sarsaType.doubleSarsa( //
        gridworld, //
        DefaultLearningRate.of(5, .51), //
        qsa1, qsa2, sac1, sac2, policy1, policy2);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures.resolve("gridworld_double_" + sarsaType + "" + nstep + ".gif"), 150, TimeUnit.MILLISECONDS)) {
      for (int index = 0; index < batches; ++index) {
        if (batches - 10 < index)
          Infoline.print(gridworld, index, ref, qsa1);
        policy.setQsa(doubleSarsa.qsa());
        policy.setSac(sac);
        ExploringStarts.batch(gridworld, policy, nstep, doubleSarsa);
        animationWriter.write(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), qsa1, ref));
      }
    }
    // qsa.print(Round.toMultipleOf(DecimalScalar.of(.01)));
    System.out.println("---");
    DiscreteVs vs = DiscreteUtils.createVs(gridworld, doubleSarsa.qsa());
    Put.of(HomeDirectory.path("gridworld_" + sarsaType), vs.values());
    Policy policyVs = PolicyType.GREEDY.bestEquiprobable(gridworld, vs, null);
    EpisodeInterface ei = EpisodeKickoff.single(gridworld, policyVs);
    while (ei.hasNext()) {
      StepRecord stepRecord = ei.step();
      Tensor state = stepRecord.prevState();
      System.out.println(state + " then " + stepRecord.action());
    }
  }

  static void main() throws Exception {
    handle(SarsaType.ORIGINAL, 1);
    handle(SarsaType.EXPECTED, 1);
    handle(SarsaType.QLEARNING, 1);
  }
}
