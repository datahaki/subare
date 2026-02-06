// code by jph
package ch.alpine.subare.book.ch04.grid;

import java.util.concurrent.TimeUnit;

import ch.alpine.subare.api.EpisodeInterface;
import ch.alpine.subare.api.LearningRate;
import ch.alpine.subare.api.Policy;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.api.StepRecord;
import ch.alpine.subare.td.Sarsa;
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

/** 1, or N-step Original/Expected Sarsa, and QLearning for gridworld
 * 
 * covers Example 4.1, p.82 */
enum Sarsa_Gridworld {
  ;
  static void handle(SarsaType sarsaType, int nstep) throws Exception {
    System.out.println(sarsaType);
    Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    int batches = 10;
    DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gridworld, qsa, sac);
    policy.setExplorationRate(LinearExplorationRate.of(batches, 0.2, 0.01));
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures.resolve("gridworld_" + sarsaType + "" + nstep + ".gif"), 250, TimeUnit.MILLISECONDS)) {
      LearningRate learningRate = DefaultLearningRate.of(2, 0.6);
      Sarsa sarsa = sarsaType.sarsa(gridworld, learningRate, qsa, sac, policy);
      for (int index = 0; index < batches; ++index) {
        animationWriter.write(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), qsa, ref));
        Infoline.print(gridworld, index, ref, qsa);
        // sarsa.supplyPolicy(() -> policy);
        ExploringStarts.batch(gridworld, policy, nstep, sarsa);
      }
    }
    // qsa.print(Round.toMultipleOf(DecimalScalar.of(.01)));
    System.out.println("---");
    DiscreteVs vs = DiscreteUtils.createVs(gridworld, qsa);
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
    int nstep = 2;
    handle(SarsaType.ORIGINAL, nstep);
    handle(SarsaType.EXPECTED, nstep);
    handle(SarsaType.QLEARNING, nstep);
  }
}
