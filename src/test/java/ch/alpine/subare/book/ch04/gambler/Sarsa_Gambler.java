// code by jph
package ch.alpine.subare.book.ch04.gambler;

import java.io.File;
import java.util.concurrent.TimeUnit;

import ch.alpine.subare.api.LearningRate;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.td.Sarsa;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.util.DefaultLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.DiscreteValueFunctions;
import ch.alpine.subare.util.EGreedyPolicy;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.LinearExplorationRate;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.subare.util.gfx.StateActionRasters;
import ch.alpine.tensor.RationalScalar;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;

/** Sarsa applied to gambler */
/* package */ class Sarsa_Gambler {
  private final GamblerModel gamblerModel;
  /** true q-function, for error measurement */
  private final DiscreteQsa ref;

  public Sarsa_Gambler(GamblerModel gamblerModel) {
    this.gamblerModel = gamblerModel;
    ref = GamblerHelper.getOptimalQsa(gamblerModel);
  }

  DiscreteQsa train(SarsaType sarsaType, int batches, LearningRate learningRate) throws Exception {
    System.out.println(sarsaType);
    GamblerRaster gamblerRaster = new GamblerRaster(gamblerModel);
    DiscreteQsa qsa = DiscreteQsa.build(gamblerModel); // q-function for training, initialized to 0
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gamblerModel, qsa, sac);
    policy.setExplorationRate(LinearExplorationRate.of(batches, 0.2, 0.01));
    // ---
    try (AnimationWriter animationWriter1 = new GifAnimationWriter(getGifFileQsa(sarsaType), 150, TimeUnit.MILLISECONDS)) {
      try (AnimationWriter animationWriter2 = new GifAnimationWriter(getGifFileSac(sarsaType), 150, TimeUnit.MILLISECONDS)) {
        Sarsa sarsa = sarsaType.sarsa(gamblerModel, learningRate, qsa, sac, policy);
        for (int index = 0; index < batches; ++index) {
          Infoline.print(gamblerModel, index, ref, qsa);
          ExploringStarts.batch(gamblerModel, policy, 1, sarsa);
          // ---
          animationWriter1.write(StateActionRasters.qsaPolicyRef(gamblerRaster, qsa, ref));
          animationWriter2.write(StateActionRasters.qsa( //
              gamblerRaster, DiscreteValueFunctions.rescaled(((DiscreteStateActionCounter) sarsa.sac()).inQsa(gamblerModel))));
        }
      }
    }
    GamblerHelper.play(gamblerModel, qsa);
    return qsa;
  }

  public static File getGifFileQsa(SarsaType sarsaType) {
    return HomeDirectory.Pictures("gambler_qsa_" + sarsaType + ".gif");
  }

  public static File getGifFileSac(SarsaType sarsaType) {
    return HomeDirectory.Pictures("gambler_sac_" + sarsaType + ".gif");
  }

  public static void main(String[] args) throws Exception {
    GamblerModel gambler = new GamblerModel(20, RationalScalar.of(4, 10));
    Sarsa_Gambler sarsa_Gambler = new Sarsa_Gambler(gambler);
    LearningRate learningRate = DefaultLearningRate.of(RealScalar.of(3), RealScalar.of(0.81));
    DiscreteQsa qsa = sarsa_Gambler.train(SarsaType.QLEARNING, 20, learningRate);
    qsa.copy();
  }
}
