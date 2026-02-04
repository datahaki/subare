// code by jph
// inspired by Shangtong Zhang
package ch.alpine.subare.book.ch08.maze;

import java.util.concurrent.TimeUnit;

import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.gfx.StateRasters;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;

/** action value iteration for cliff walk */
enum AVH_Dynamaze {
  ;
  public static void create(String name, Dynamaze dynamaze) throws Exception {
    DiscreteQsa est = DynamazeHeuristic.create(dynamaze);
    // est = DiscreteQsa.build(dynamaze);
    ActionValueIteration avi = ActionValueIteration.of(dynamaze, est);
    // ---
    DiscreteQsa ref = DynamazeHelper.getOptimalQsa(dynamaze);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures(name + "_qsa_avi.gif"), 500, TimeUnit.MILLISECONDS)) {
      DynamazeRaster dynamazeRaster = new DynamazeRaster(dynamaze);
      for (int index = 0; index < 50; ++index) {
        Infoline infoline = Infoline.print(dynamaze, index, ref, avi.qsa());
        animationWriter.write(StateRasters.qsaLossRef(dynamazeRaster, avi.qsa(), ref));
        avi.step();
        if (infoline.isLossfree())
          break;
      }
    }
  }

  static void main() throws Exception {
    // create("maze2", DynamazeHelper.original("maze2"));
    create("maze5", DynamazeHelper.create5(2));
  }
}
