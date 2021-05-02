// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import java.util.concurrent.TimeUnit;

import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;
import ch.ethz.idsc.subare.core.alg.Random1StepTabularQPlanning;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.TabularSteps;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;

/** Example 4.1, p.82 */
enum RSTQP_Gridworld {
  ;
  public static void main(String[] args) throws Exception {
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
