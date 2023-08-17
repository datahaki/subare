// code by jph
// inspired by Shangtong Zhang
package ch.alpine.subare.demo.fish;

import java.util.concurrent.TimeUnit;

import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.td.Sarsa;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.util.DefaultLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.EGreedyPolicy;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.LinearExplorationRate;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.subare.util.gfx.StateRasters;
import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;
import ch.alpine.tensor.sca.Round;

/** StepDigest qsa methods applied to cliff walk */
enum Sarsa_Fishfarm {
  ;
  static void handle(SarsaType sarsaType, int nstep, int batches) throws Exception {
    System.out.println(sarsaType);
    Fishfarm fishfarm = new Fishfarm(20, 20);
    FishfarmRaster fishfarmRaster = new FishfarmRaster(fishfarm);
    final DiscreteQsa ref = FishfarmHelper.getOptimalQsa(fishfarm);
    DiscreteQsa qsa = DiscreteQsa.build(fishfarm, DoubleScalar.POSITIVE_INFINITY);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(fishfarm, qsa, sac);
    policy.setExplorationRate(LinearExplorationRate.of(batches, 0.5, 0.01));
    Sarsa sarsa = sarsaType.sarsa(fishfarm, DefaultLearningRate.of(7, 0.61), qsa, sac, policy);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures("fishfarm_qsa_" + sarsaType + ".gif"), 200, TimeUnit.MILLISECONDS)) {
      for (int index = 0; index < batches; ++index) {
        // if (batches - 10 < index)
        Infoline infoline = Infoline.print(fishfarm, index, ref, qsa);
        // sarsa.supplyPolicy(() -> policy);
        ExploringStarts.batch(fishfarm, policy, nstep, sarsa);
        animationWriter.write(StateRasters.qsaLossRef(fishfarmRaster, qsa, ref));
        if (infoline.isLossfree())
          break;
      }
    }
    DiscreteUtils.print(qsa, Round._2);
    // System.out.println("---");
    // Policy policy = GreedyPolicy.bestEquiprobable(cliffwalk, qsa);
    // EpisodeInterface mce = EpisodeKickoff.single(cliffwalk, policy);
    // while (mce.hasNext()) {
    // StepInterface stepInterface = mce.step();
    // Tensor state = stepInterface.prevState();
    // System.out.println(state + " then " + stepInterface.action());
    // }
  }

  public static void main(String[] args) throws Exception {
    // handle(SarsaType.original, 1, 30);
    handle(SarsaType.EXPECTED, 1, 30);
    // handle(SarsaType.qlearning, 1, 10);
  }
}
