// code by jph
// inspired by Shangtong Zhang
package ch.alpine.subare.book.ch06.windy;

import java.nio.file.Path;
import java.util.concurrent.TimeUnit;

import ch.alpine.subare.api.LearningRate;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.td.Sarsa;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.util.DefaultLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.EGreedyPolicy;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.subare.util.gfx.StateActionRasters;
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

  public static Path getFileQsa(SarsaType sarsaType) {
    return HomeDirectory.Pictures.resolve("windygrid_qsa_" + sarsaType + ".gif");
  }

  static void main() throws Exception {
    // handle(SarsaType.original, 20);
    // handle(SarsaType.expected, 20);
    handle(SarsaType.QLEARNING, 20);
  }
}
