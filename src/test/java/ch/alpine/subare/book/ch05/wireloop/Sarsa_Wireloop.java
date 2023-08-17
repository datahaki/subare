// code by jph
package ch.alpine.subare.book.ch05.wireloop;

import java.util.concurrent.TimeUnit;

import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.td.Sarsa;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.util.DefaultLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.EGreedyPolicy;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.LinearExplorationRate;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;

enum Sarsa_Wireloop {
  ;
  static void handle(SarsaType sarsaType, int nstep, int batches) throws Exception {
    System.out.println(sarsaType);
    String name = "wire5";
    WireloopReward wireloopReward = WireloopReward.freeSteps();
    wireloopReward = WireloopReward.constantCost();
    Wireloop wireloop = WireloopHelper.create(name, WireloopReward::id_x, wireloopReward);
    WireloopRaster wireloopRaster = new WireloopRaster(wireloop);
    DiscreteQsa ref = WireloopHelper.getOptimalQsa(wireloop);
    DiscreteQsa qsa = DiscreteQsa.build(wireloop);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(wireloop, qsa, sac);
    policy.setExplorationRate(LinearExplorationRate.of(batches, 0.2, 0.01));
    Sarsa sarsa = sarsaType.sarsa(wireloop, DefaultLearningRate.of(3, 0.51), qsa, sac, policy);
    try (AnimationWriter animationWriter = new GifAnimationWriter( //
        HomeDirectory.Pictures(name + "L_qsa_" + sarsaType + "" + nstep + ".gif"), 250, TimeUnit.MILLISECONDS)) {
      for (int index = 0; index < batches; ++index) {
        Infoline infoline = Infoline.print(wireloop, index, ref, qsa);
        ExploringStarts.batch(wireloop, policy, nstep, sarsa);
        animationWriter.write(WireloopHelper.render(wireloopRaster, ref, qsa));
        if (infoline.isLossfree())
          break;
      }
    }
    System.out.println("---");
  }

  public static void main(String[] args) throws Exception {
    handle(SarsaType.QLEARNING, 1, 20);
    handle(SarsaType.EXPECTED, 1, 20);
  }
}
