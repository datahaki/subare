// code by jph
package ch.alpine.subare.book.ch04.grid;

import java.util.concurrent.TimeUnit;

import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.mc.MonteCarloExploringStarts;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.EGreedyPolicy;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.LinearExplorationRate;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.subare.util.gfx.StateActionRasters;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;

/** Example 4.1, p.82 */
enum MCES_Gridworld {
  ;
  static void main() throws Exception {
    Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(gridworld);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures.resolve("gridworld_qsa_mces.gif"), 250, TimeUnit.MILLISECONDS)) {
      final int batches = 20;
      StateActionCounter sac = new DiscreteStateActionCounter();
      EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gridworld, mces.qsa(), sac);
      policy.setExplorationRate(LinearExplorationRate.of(batches, 0.2, 0.05));
      for (int index = 0; index < batches; ++index) {
        Infoline.print(gridworld, index, ref, mces.qsa());
        for (int count = 0; count < 1; ++count) {
          ExploringStarts.batch(gridworld, policy, mces);
        }
        animationWriter.write(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), mces.qsa(), ref));
      }
    }
  }
}
