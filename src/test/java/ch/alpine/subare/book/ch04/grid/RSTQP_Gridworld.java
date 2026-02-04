// code by jph
package ch.alpine.subare.book.ch04.grid;

import java.util.concurrent.TimeUnit;

import ch.alpine.subare.alg.Random1StepTabularQPlanning;
import ch.alpine.subare.util.ConstantLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.TabularSteps;
import ch.alpine.subare.util.gfx.StateActionRasters;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;

/** Example 4.1, p.82 */
enum RSTQP_Gridworld {
  ;
  static void main() throws Exception {
    Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    Random1StepTabularQPlanning rstqp = Random1StepTabularQPlanning.of( //
        gridworld, qsa, ConstantLearningRate.one());
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures("gridworld_qsa_rstqp.gif"), 250, TimeUnit.MILLISECONDS)) {
      int batches = 10;
      for (int index = 0; index < batches; ++index) {
        animationWriter.write(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), qsa, ref));
        Infoline.print(gridworld, index, ref, qsa);
        TabularSteps.batch(gridworld, gridworld, rstqp);
      }
    }
  }
}
