// code by jph
// inspired by Shangtong Zhang
package ch.alpine.subare.book.ch06.windy;

import java.util.concurrent.TimeUnit;

import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.api.Policy;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteUtils;
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
enum AVI_Windygrid {
  ;
  public static void main(String[] args) throws Exception {
    Windygrid windygrid = Windygrid.createFour();
    WindygridRaster windygridRaster = new WindygridRaster(windygrid);
    DiscreteQsa ref = WindygridHelper.getOptimalQsa(windygrid);
    Export.of(HomeDirectory.Pictures("windygrid_qsa_avi.png"), //
        StateActionRasters.qsa_rescaled(windygridRaster, ref));
    ActionValueIteration avi = ActionValueIteration.of(windygrid);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures("windygrid_qsa_avi.gif"), 250, TimeUnit.MILLISECONDS)) {
      for (int index = 0; index < 20; ++index) {
        Infoline infoline = Infoline.print(windygrid, index, ref, avi.qsa());
        animationWriter.write(StateActionRasters.qsaLossRef(windygridRaster, avi.qsa(), ref));
        avi.step();
        if (infoline.isLossfree())
          break;
      }
    }
    // TODO SUBARE extract code below to other file
    DiscreteVs vs = DiscreteUtils.createVs(windygrid, ref);
    DiscreteUtils.print(vs);
    Policy policy = PolicyType.GREEDY.bestEquiprobable(windygrid, ref, null);
    Policies.print(policy, windygrid.states());
  }
}
