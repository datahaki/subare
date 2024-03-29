// code by jph
// inspired by Shangtong Zhang
package ch.alpine.subare.book.ch06.cliff;

import java.util.concurrent.TimeUnit;

import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.api.Policy;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.DiscreteValueFunctions;
import ch.alpine.subare.util.DiscreteVs;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.Policies;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.subare.util.gfx.StateActionRasters;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.Export;
import ch.alpine.tensor.io.GifAnimationWriter;

/** action value iteration for cliff walk */
enum AVI_Cliffwalk {
  ;
  public static void main(String[] args) throws Exception {
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    CliffwalkRaster cliffwalkRaster = new CliffwalkRaster(cliffwalk);
    DiscreteQsa ref = CliffwalkHelper.getOptimalQsa(cliffwalk);
    Export.of(HomeDirectory.Pictures("cliffwalk_qsa_avi.png"), //
        StateActionRasters.qsa(new CliffwalkRaster(cliffwalk), DiscreteValueFunctions.rescaled(ref)));
    ActionValueIteration avi = ActionValueIteration.of(cliffwalk);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures("cliffwalk_qsa_avi.gif"), 200, TimeUnit.MILLISECONDS)) {
      for (int index = 0; index < 20; ++index) {
        Infoline infoline = Infoline.print(cliffwalk, index, ref, avi.qsa());
        animationWriter.write(StateActionRasters.qsaLossRef(cliffwalkRaster, avi.qsa(), ref));
        avi.step();
        if (infoline.isLossfree())
          break;
      }
      animationWriter.write(StateActionRasters.qsaLossRef(cliffwalkRaster, avi.qsa(), ref));
    }
    DiscreteVs vs = DiscreteUtils.createVs(cliffwalk, ref);
    DiscreteUtils.print(vs);
    Policy policy = PolicyType.GREEDY.bestEquiprobable(cliffwalk, ref, null);
    Policies.print(policy, cliffwalk.states());
  }
}
