// code by jph
package ch.alpine.subare.book.ch04.gambler;

import java.util.concurrent.TimeUnit;

import ch.alpine.subare.api.FeatureMapper;
import ch.alpine.subare.api.LearningRate;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.td.TrueOnlineSarsa;
import ch.alpine.subare.util.DefaultLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.EGreedyPolicy;
import ch.alpine.subare.util.ExactFeatureMapper;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.FeatureWeight;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.subare.util.gfx.StateActionRasters;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.ext.Timing;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;

/* package */ enum TOS_Gambler {
  ;
  private static final Scalar LAMBDA = RealScalar.of(0.3);

  static void run(SarsaType sarsaType) throws Exception {
    System.out.println(sarsaType);
    GamblerModel gamblerModel = new GamblerModel(20, RealScalar.of(.4));
    final DiscreteQsa ref = GamblerHelper.getOptimalQsa(gamblerModel);
    FeatureMapper mapper = ExactFeatureMapper.of(gamblerModel);
    FeatureWeight w = new FeatureWeight(mapper);
    DiscreteQsa qsa = DiscreteQsa.build(gamblerModel);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gamblerModel, qsa, sac);
    LearningRate learningRate = DefaultLearningRate.of(RealScalar.of(3), RealScalar.of(0.81));
    // LearningRate learningRate = ConstantLearningRate.of(RealScalar.of(0.3), false); // the case without warmStart
    TrueOnlineSarsa trueOnlineSarsa = sarsaType.trueOnline(gamblerModel, LAMBDA, mapper, learningRate, w, sac, policy);
    final String name = sarsaType.name().toLowerCase();
    Timing timing = Timing.started();
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures.resolve("gambler_tos_" + name + ".gif"), 250, TimeUnit.MILLISECONDS)) {
      for (int batch = 0; batch < 100; ++batch) {
        // System.out.println("batch " + batch);
        policy.setQsa(trueOnlineSarsa.qsaInterface());
        ExploringStarts.batch(gamblerModel, policy, trueOnlineSarsa);
        // DiscreteQsa toQsa = trueOnlineSarsa.getQsa();
        // XYtoSarsa.append(Tensors.vector(RealScalar.of(index).number(), errorAnalysis.getError(monteCarloInterface, optimalQsa, toQsa).number()));
        DiscreteQsa qsaRef = trueOnlineSarsa.qsa();
        Infoline infoline = Infoline.print(gamblerModel, batch, ref, qsaRef);
        animationWriter.write(StateActionRasters.qsaLossRef(new GamblerRaster(gamblerModel), qsaRef, ref));
        if (infoline.isLossfree()) {
          animationWriter.write(StateActionRasters.qsaLossRef(new GamblerRaster(gamblerModel), qsaRef, ref));
          animationWriter.write(StateActionRasters.qsaLossRef(new GamblerRaster(gamblerModel), qsaRef, ref));
          animationWriter.write(StateActionRasters.qsaLossRef(new GamblerRaster(gamblerModel), qsaRef, ref));
          break;
        }
      }
    }
    System.out.println("Time for TrueOnlineSarsa: " + timing.seconds() + "s");
  }

  static void main() throws Exception {
    run(SarsaType.ORIGINAL);
    run(SarsaType.EXPECTED);
    run(SarsaType.QLEARNING);
  }
}
