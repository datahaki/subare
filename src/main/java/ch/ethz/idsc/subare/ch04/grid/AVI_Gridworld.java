// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch04.grid;

import java.util.concurrent.TimeUnit;

import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;
import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteValueFunctions;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;

/** solving grid world
 * gives the value function for the optimal policy equivalent to
 * shortest path to terminal state
 * 
 * Example 4.1, p.82
 * 
 * {0, 0} 0
 * {0, 1} -1
 * {0, 2} -2
 * {0, 3} -3
 * {1, 0} -1
 * {1, 1} -2
 * {1, 2} -3
 * {1, 3} -2
 * {2, 0} -2
 * {2, 1} -3
 * {2, 2} -2
 * {2, 3} -1
 * {3, 0} -3
 * {3, 1} -2
 * {3, 2} -1
 * {3, 3} 0 */
enum AVI_Gridworld {
  ;
  public static void main(String[] args) throws Exception {
    Gridworld gridworld = new Gridworld();
    GridworldRaster gridworldRaster = new GridworldRaster(gridworld);
    ActionValueIteration avi = ActionValueIteration.of(gridworld);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures("gridworld_qsa_avi.gif"), 250, TimeUnit.MILLISECONDS)) {
      for (int count = 0; count < 7; ++count) {
        animationWriter.write(StateActionRasters.qsa(gridworldRaster, DiscreteValueFunctions.rescaled(avi.qsa())));
        avi.step();
      }
      animationWriter.write(StateActionRasters.qsa(gridworldRaster, DiscreteValueFunctions.rescaled(avi.qsa())));
    }
    // ---
    DiscreteUtils.print(avi.qsa());
    DiscreteVs dvs = DiscreteUtils.createVs(gridworld, avi.qsa());
    DiscreteUtils.print(dvs);
  }
}
