// code by jph
package ch.alpine.subare.book.ch06.cliff;

import java.util.concurrent.TimeUnit;

import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.mc.MonteCarloExploringStarts;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.PolicyBase;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.subare.util.gfx.StateActionRasters;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;

/** monte carlo is bad in this example, since the steep negative reward biases most episodes */
// TODO SUBARE this does not really converge at all
enum MCES_Cliffwalk {
  ;
  public static void main(String[] args) throws Exception {
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    CliffwalkRaster cliffwalkRaster = new CliffwalkRaster(cliffwalk);
    final DiscreteQsa ref = CliffwalkHelper.getOptimalQsa(cliffwalk);
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(cliffwalk);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures("cliffwalk_qsa_mces.gif"), 100, TimeUnit.MILLISECONDS)) {
      int batches = 100;
      for (int index = 0; index < batches; ++index) {
        Infoline.print(cliffwalk, index, ref, mces.qsa());
        for (int count = 0; count < 10; ++count) {
          StateActionCounter sac = new DiscreteStateActionCounter();
          PolicyBase policy = PolicyType.EGREEDY.bestEquiprobable(cliffwalk, mces.qsa(), sac);
          ExploringStarts.batch(cliffwalk, policy, mces);
        }
        animationWriter.write(StateActionRasters.qsaLossRef(cliffwalkRaster, mces.qsa(), ref));
      }
    }
  }
}
