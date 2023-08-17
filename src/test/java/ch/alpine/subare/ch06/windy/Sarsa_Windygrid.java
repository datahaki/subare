// code by jph
// inspired by Shangtong Zhang
package ch.alpine.subare.ch06.windy;

import java.io.File;
import java.util.concurrent.TimeUnit;

import ch.alpine.subare.core.api.StateActionCounter;
import ch.alpine.subare.core.td.Sarsa;
import ch.alpine.subare.core.td.SarsaType;
import ch.alpine.subare.core.util.DefaultLearningRate;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.subare.core.util.DiscreteStateActionCounter;
import ch.alpine.subare.core.util.EGreedyPolicy;
import ch.alpine.subare.core.util.ExploringStarts;
import ch.alpine.subare.core.util.Infoline;
import ch.alpine.subare.core.util.LearningRate;
import ch.alpine.subare.core.util.PolicyType;
import ch.alpine.subare.core.util.gfx.StateActionRasters;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;

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
        new GifAnimationWriter(getFileQsa(sarsaType), 100, TimeUnit.MILLISECONDS)) {
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

  public static File getFileQsa(SarsaType sarsaType) {
    return HomeDirectory.Pictures("windygrid_qsa_" + sarsaType + ".gif");
  }

  public static void main(String[] args) throws Exception {
    // handle(SarsaType.original, 20);
    // handle(SarsaType.expected, 20);
    handle(SarsaType.QLEARNING, 20);
  }
}
