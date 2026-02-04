// code by jph
// inspired by Shangtong Zhang
package ch.alpine.subare.book.ch08.maze;

import java.util.concurrent.TimeUnit;

import ch.alpine.subare.api.LearningRate;
import ch.alpine.subare.api.Policy;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.td.PrioritizedSweeping;
import ch.alpine.subare.td.Sarsa;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.util.DefaultLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.EGreedyPolicy;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.LinearExplorationRate;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.subare.util.StepExploringStarts;
import ch.alpine.subare.util.gfx.StateRasters;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;

/** determines q(s, a) function for equiprobable "random" policy */
enum PS_Dynamaze {
  ;
  static void handle(SarsaType sarsaType, int batches) throws Exception {
    System.out.println(sarsaType);
    String name = "maze2";
    Dynamaze dynamaze;
    // dynamaze = DynamazeHelper.original(name);
    dynamaze = DynamazeHelper.create5(3);
    DynamazeRaster dynamazeRaster = new DynamazeRaster(dynamaze);
    final DiscreteQsa ref = DynamazeHelper.getOptimalQsa(dynamaze);
    DiscreteQsa qsa = DiscreteQsa.build(dynamaze);
    LearningRate learningRate = DefaultLearningRate.of(7, 1.01);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(dynamaze, qsa, sac);
    policy.setExplorationRate(LinearExplorationRate.of(batches, 0.1, 0.01));
    Sarsa sarsa = sarsaType.sarsa(dynamaze, learningRate, qsa, sac, policy);
    PrioritizedSweeping prioritizedSweeping = new PrioritizedSweeping( //
        sarsa, 10, RealScalar.ZERO);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures(name + "_ps_" + sarsaType + ".gif"), 250, TimeUnit.MILLISECONDS)) {
      StepExploringStarts stepExploringStarts = //
          new StepExploringStarts(dynamaze, prioritizedSweeping) {
            @Override
            public Policy batchPolicy(int batch) {
              return policy;
            }
          };
      while (stepExploringStarts.batchIndex() < batches) {
        Infoline infoline = Infoline.print(dynamaze, stepExploringStarts.batchIndex(), ref, qsa);
        stepExploringStarts.nextEpisode();
        animationWriter.write(StateRasters.qsaLossRef(dynamazeRaster, qsa, ref));
        if (infoline.isLossfree())
          break;
      }
    }
  }

  static void main() throws Exception {
    // handle(SarsaType.original, 10);
    // handle(SarsaType.expected, 50);
    handle(SarsaType.QLEARNING, 10);
  }
}
