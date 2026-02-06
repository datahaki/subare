// code by jph
package ch.alpine.subare.book.ch04.grid;

import java.util.concurrent.TimeUnit;

import ch.alpine.subare.api.LearningRate;
import ch.alpine.subare.api.Policy;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.td.Sarsa;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.util.DefaultLearningRate;
import ch.alpine.subare.util.DequeExploringStarts;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.EGreedyPolicy;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.LinearExplorationRate;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.subare.util.gfx.StateActionRasters;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;

/** 1, or N-step Original/Expected Sarsa, and QLearning for gridworld
 * 
 * covers Example 4.1, p.82 */
enum SES_Gridworld {
  ;
  static void handle(SarsaType sarsaType, int nstep, int batches) throws Exception {
    System.out.println(sarsaType);
    Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gridworld, qsa, sac);
    policy.setExplorationRate(LinearExplorationRate.of(batches, 0.2, 0.01));
    try (AnimationWriter animationWriter = new GifAnimationWriter( //
        HomeDirectory.Pictures.resolve("gridworld_ses_" + sarsaType + "" + nstep + ".gif"), 250, TimeUnit.MILLISECONDS)) {
      LearningRate learningRate = DefaultLearningRate.of(5, 1.1);
      Sarsa sarsa = sarsaType.sarsa(gridworld, learningRate, qsa, sac, policy);
      DequeExploringStarts exploringStartsStream = new DequeExploringStarts(gridworld, nstep, sarsa) {
        @Override
        public Policy batchPolicy(int batch) {
          return policy;
        }
      };
      int episode = 0;
      while (exploringStartsStream.batchIndex() < batches) {
        exploringStartsStream.nextEpisode();
        if (episode % 5 == 0) {
          Infoline infoline = Infoline.print(gridworld, episode, ref, qsa);
          animationWriter.write(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), qsa, ref));
          if (infoline.isLossfree())
            break;
        }
        ++episode;
      }
    }
  }

  static void main() throws Exception {
    int nstep = 1;
    handle(SarsaType.ORIGINAL, nstep, 3);
    handle(SarsaType.EXPECTED, nstep, 3);
    handle(SarsaType.QLEARNING, nstep, 3);
  }
}
