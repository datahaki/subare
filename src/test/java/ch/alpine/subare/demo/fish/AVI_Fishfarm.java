// code by jph
// inspired by Shangtong Zhang
package ch.alpine.subare.demo.fish;

import java.util.concurrent.TimeUnit;

import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.gfx.StateRasters;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;

/** action value iteration for cliff walk */
enum AVI_Fishfarm {
  ;
  public static void main(String[] args) throws Exception {
    Fishfarm fishfarm = new Fishfarm(20, 20);
    FishfarmRaster fishfarmRaster = new FishfarmRaster(fishfarm);
    DiscreteQsa ref = FishfarmHelper.getOptimalQsa(fishfarm);
    // Export.of(UserHome.Pictures("cliffwalk_qsa_avi.png"), //
    // StateActionRasters.qsa(new CliffwalkRaster(cliffwalk), DiscreteValueFunctions.rescaled(ref)));
    ActionValueIteration avi = ActionValueIteration.of(fishfarm);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures("fishfarm_qsa_avi.gif"), 200, TimeUnit.MILLISECONDS)) {
      for (int index = 0; index < 20; ++index) {
        Infoline infoline = Infoline.print(fishfarm, index, ref, avi.qsa());
        animationWriter.write(StateRasters.qsaLossRef(fishfarmRaster, avi.qsa(), ref));
        avi.step();
        if (infoline.isErrorFree())
          break;
      }
      animationWriter.write(StateRasters.qsaLossRef(fishfarmRaster, avi.qsa(), ref));
      animationWriter.write(StateRasters.qsaLossRef(fishfarmRaster, avi.qsa(), ref));
      animationWriter.write(StateRasters.qsaLossRef(fishfarmRaster, avi.qsa(), ref));
    }
    // DiscreteVs vs = DiscreteUtils.createVs(cliffwalk, ref);
    // vs.print();
    // Policy policy = GreedyPolicy.bestEquiprobable(cliffwalk, ref);
    // Policies.print(policy, cliffwalk.states());
  }
}
