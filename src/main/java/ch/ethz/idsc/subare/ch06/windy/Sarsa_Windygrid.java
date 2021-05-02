// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch06.windy;

import java.util.concurrent.TimeUnit;

import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteStateActionCounter;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.subare.core.util.PolicyType;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;

/** determines q(s, a) function for equiprobable "random" policy */
enum Sarsa_Windygrid {
  ;
  static void handle(SarsaType sarsaType, int batches) throws Exception {
    System.out.println(sarsaType);
    Windygrid windygrid = Windygrid.createFour();
    WindygridRaster windygridRaster = new WindygridRaster(windygrid);
    final DiscreteQsa ref = WindygridHelper.getOptimalQsa(windygrid);
    DiscreteQsa qsa = DiscreteQsa.build(windygrid);
    LearningRate learningRate = DefaultLearningRate.of(3, 0.51);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(windygrid, qsa, sac);
    Sarsa sarsa = sarsaType.sarsa(windygrid, learningRate, qsa, sac, policy);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures("windygrid_qsa_" + sarsaType + ".gif"), 100, TimeUnit.MILLISECONDS)) {
      for (int index = 0; index < batches; ++index) {
        Infoline infoline = Infoline.print(windygrid, index, ref, qsa);
        // sarsa.supplyPolicy(() -> policy);
        for (int count = 0; count < 10; ++count) // because there is only 1 start state
          ExploringStarts.batch(windygrid, policy, sarsa);
        animationWriter.write(StateActionRasters.qsaLossRef(windygridRaster, qsa, ref));
        if (infoline.isLossfree())
          break;
      }
    }
  }

  public static void main(String[] args) throws Exception {
    // handle(SarsaType.original, 20);
    // handle(SarsaType.expected, 20);
    handle(SarsaType.QLEARNING, 20);
  }
}
