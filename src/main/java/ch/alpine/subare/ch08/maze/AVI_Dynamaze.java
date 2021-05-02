// code by jph
// inspired by Shangtong Zhang
package ch.alpine.subare.ch08.maze;

import java.util.concurrent.TimeUnit;

import ch.alpine.subare.core.Policy;
import ch.alpine.subare.core.alg.ActionValueIteration;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.subare.core.util.DiscreteUtils;
import ch.alpine.subare.core.util.DiscreteVs;
import ch.alpine.subare.core.util.Infoline;
import ch.alpine.subare.core.util.Policies;
import ch.alpine.subare.core.util.PolicyType;
import ch.alpine.subare.core.util.gfx.StateRasters;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;

/** action value iteration for cliff walk */
enum AVI_Dynamaze {
  ;
  public static void main(String[] args) throws Exception {
    String name = "maze5";
    Dynamaze dynamaze = DynamazeHelper.create5(3);
    DynamazeRaster dynamazeRaster = new DynamazeRaster(dynamaze);
    DiscreteQsa ref = DynamazeHelper.getOptimalQsa(dynamaze);
    // Export.of(UserHome.Pictures("dynamaze_qsa_avi.png"), //
    // DynamazeHelper.render(windygrid, DiscreteValueFunctions.rescaled(ref)));
    ActionValueIteration avi = ActionValueIteration.of(dynamaze);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures(name + "_qsa_avi.gif"), 250, TimeUnit.MILLISECONDS)) {
      for (int index = 0; index < 50; ++index) {
        Infoline infoline = Infoline.print(dynamaze, index, ref, avi.qsa());
        animationWriter.write(StateRasters.qsaLossRef(dynamazeRaster, avi.qsa(), ref));
        avi.step();
        if (infoline.isLossfree())
          break;
      }
      // gsw.append(ImageFormat.of(DynamazeHelper.render(dynamaze, avi.qsa(), ref)));
    }
    // TODO extract code below to other file
    DiscreteVs vs = DiscreteUtils.createVs(dynamaze, ref);
    DiscreteUtils.print(vs);
    Policy policy = PolicyType.GREEDY.bestEquiprobable(dynamaze, ref, null);
    Policies.print(policy, dynamaze.states());
  }
}
