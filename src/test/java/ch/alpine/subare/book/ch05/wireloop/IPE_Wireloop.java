// code by jph
package ch.alpine.subare.book.ch05.wireloop;

import java.util.concurrent.TimeUnit;

import ch.alpine.subare.alg.IterativePolicyEvaluation;
import ch.alpine.subare.api.Policy;
import ch.alpine.subare.util.EquiprobablePolicy;
import ch.alpine.subare.util.gfx.StateRasters;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;

enum IPE_Wireloop {
  ;
  public static void main(String[] args) throws Exception {
    String name = "wire5";
    Wireloop wireloop = WireloopHelper.create(name, WireloopReward::id_x);
    WireloopRaster wireloopRaster = new WireloopRaster(wireloop);
    Policy policy = EquiprobablePolicy.create(wireloop);
    IterativePolicyEvaluation ipe = new IterativePolicyEvaluation( //
        wireloop, policy);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures(name + "_ipe_iteration.gif"), 200, TimeUnit.MILLISECONDS)) {
      for (int count = 0; count < 20; ++count) {
        System.out.println(count);
        animationWriter.write(StateRasters.vs_rescale(wireloopRaster, ipe.vs()));
        for (int ep = 0; ep < 5; ++ep)
          ipe.step();
      }
    }
  }
}
