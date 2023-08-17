// code by jph
package ch.alpine.subare.book.ch05.wireloop;

import java.util.concurrent.TimeUnit;

import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.mc.MonteCarloExploringStarts;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.EGreedyPolicy;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.LinearExplorationRate;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;

enum MCES_Wireloop {
  ;
  public static void main(String[] args) throws Exception {
    String name = "wire5";
    Wireloop wireloop = WireloopHelper.create(name, WireloopReward::id_x);
    WireloopRaster wireloopRaster = new WireloopRaster(wireloop);
    DiscreteQsa ref = WireloopHelper.getOptimalQsa(wireloop);
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(wireloop);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures(name + "L_mces.gif"), 100, TimeUnit.MILLISECONDS)) {
      int batches = 10;
      StateActionCounter sac = new DiscreteStateActionCounter();
      EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(wireloop, mces.qsa(), sac);
      policy.setExplorationRate(LinearExplorationRate.of(batches, 0.2, 0.05));
      for (int index = 0; index < batches; ++index) {
        Infoline infoline = Infoline.print(wireloop, index, ref, mces.qsa());
        for (int count = 0; count < 4; ++count) {
          ExploringStarts.batch(wireloop, policy, mces);
        }
        animationWriter.write(WireloopHelper.render(wireloopRaster, ref, mces.qsa()));
        if (infoline.isLossfree())
          break;
      }
    }
  }
}
