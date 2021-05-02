// code by jph
// inspired by Shangtong Zhang
package ch.alpine.subare.ch08.maze;

import java.util.concurrent.TimeUnit;

import ch.alpine.subare.core.StateActionCounter;
import ch.alpine.subare.core.td.Sarsa;
import ch.alpine.subare.core.td.SarsaType;
import ch.alpine.subare.core.td.TabularDynaQ;
import ch.alpine.subare.core.util.DefaultLearningRate;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.subare.core.util.DiscreteStateActionCounter;
import ch.alpine.subare.core.util.EGreedyPolicy;
import ch.alpine.subare.core.util.ExploringStarts;
import ch.alpine.subare.core.util.Infoline;
import ch.alpine.subare.core.util.LearningRate;
import ch.alpine.subare.core.util.LinearExplorationRate;
import ch.alpine.subare.core.util.PolicyType;
import ch.alpine.subare.core.util.gfx.StateRasters;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;

/** determines q(s, a) function for equiprobable "random" policy */
enum TDQ_Dynamaze {
  ;
  static void handle(SarsaType sarsaType, int batches) throws Exception {
    System.out.println(sarsaType);
    String name = "maze5";
    Dynamaze dynamaze = DynamazeHelper.create5(3);
    DynamazeRaster dynamazeRaster = new DynamazeRaster(dynamaze);
    final DiscreteQsa ref = DynamazeHelper.getOptimalQsa(dynamaze);
    DiscreteQsa qsa = DiscreteQsa.build(dynamaze);
    LearningRate learningRate = DefaultLearningRate.of(5, 0.51);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(dynamaze, qsa, sac);
    policy.setExplorationRate(LinearExplorationRate.of(batches, 0.2, 0.01));
    Sarsa sarsa = sarsaType.sarsa(dynamaze, learningRate, qsa, sac, policy);
    TabularDynaQ tabularDynaQ = new TabularDynaQ(sarsa, 10);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures(name + "_tdq_" + sarsaType + ".gif"), 200, TimeUnit.MILLISECONDS)) {
      for (int index = 0; index < batches; ++index) {
        // if (EPISODES - 10 < index)
        Infoline.print(dynamaze, index, ref, qsa);
        // for (int count = 0; count < 5; ++count)
        ExploringStarts.batch(dynamaze, policy, tabularDynaQ);
        animationWriter.write(StateRasters.vs_rescale(dynamazeRaster, qsa));
      }
    }
  }

  public static void main(String[] args) throws Exception {
    // handle(SarsaType.original, 3, 50);
    // handle(SarsaType.expected, 2, 50);
    handle(SarsaType.QLEARNING, 50);
  }
}
