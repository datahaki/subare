// code by jph
package ch.alpine.subare.ch08.maze;

import java.util.concurrent.TimeUnit;

import ch.alpine.subare.core.alg.Random1StepTabularQPlanning;
import ch.alpine.subare.core.util.ConstantLearningRate;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.subare.core.util.Infoline;
import ch.alpine.subare.core.util.TabularSteps;
import ch.alpine.subare.core.util.gfx.StateRasters;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;

/**  */
enum RSTQP_Dynamaze {
  ;
  public static void main(String[] args) throws Exception {
    String name = "maze5";
    Dynamaze dynamaze = DynamazeHelper.create5(3);
    DynamazeRaster dynamazeRaster = new DynamazeRaster(dynamaze);
    DiscreteQsa ref = DynamazeHelper.getOptimalQsa(dynamaze);
    DiscreteQsa qsa = DiscreteQsa.build(dynamaze);
    Random1StepTabularQPlanning rstqp = Random1StepTabularQPlanning.of( //
        dynamaze, qsa, ConstantLearningRate.of(RealScalar.ONE));
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures(name + "_qsa_rstqp.gif"), 250, TimeUnit.MILLISECONDS)) {
      int batches = 50;
      for (int index = 0; index < batches; ++index) {
        Infoline infoline = Infoline.print(dynamaze, index, ref, qsa);
        TabularSteps.batch(dynamaze, dynamaze, rstqp);
        animationWriter.write(StateRasters.vs_rescale(dynamazeRaster, qsa));
        if (infoline.isLossfree())
          break;
      }
    }
  }
}
