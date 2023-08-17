// code by jph
package ch.alpine.subare.book.ch05.wireloop;

import java.util.concurrent.TimeUnit;

import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.Infoline;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;

enum AVI_Wireloop {
  ;
  public static void main(String[] args) throws Exception {
    String name = "wirec";
    WireloopReward wireloopReward = WireloopReward.freeSteps();
    wireloopReward = WireloopReward.constantCost();
    Wireloop wireloop = WireloopHelper.create(name, WireloopReward::id_x, wireloopReward);
    WireloopRaster wireloopRaster = new WireloopRaster(wireloop);
    DiscreteQsa ref = WireloopHelper.getOptimalQsa(wireloop);
    ActionValueIteration avi = ActionValueIteration.of(wireloop);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures(name + "L_avi.gif"), 250, TimeUnit.MILLISECONDS)) {
      int batches = 50;
      for (int index = 0; index < batches; ++index) {
        Infoline infoline = Infoline.print(wireloop, index, ref, avi.qsa());
        animationWriter.write(WireloopHelper.render(wireloopRaster, ref, avi.qsa()));
        avi.step();
        if (infoline.isLossfree())
          break;
      }
      animationWriter.write(WireloopHelper.render(wireloopRaster, ref, avi.qsa()));
    }
  }
}
