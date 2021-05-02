// code by jph
package ch.alpine.subare.ch04.gambler;

import java.util.concurrent.TimeUnit;

import ch.alpine.subare.core.StateActionCounter;
import ch.alpine.subare.core.td.SarsaType;
import ch.alpine.subare.core.td.TrueOnlineSarsa;
import ch.alpine.subare.core.util.DefaultLearningRate;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.subare.core.util.DiscreteStateActionCounter;
import ch.alpine.subare.core.util.EGreedyPolicy;
import ch.alpine.subare.core.util.ExactFeatureMapper;
import ch.alpine.subare.core.util.ExploringStarts;
import ch.alpine.subare.core.util.FeatureMapper;
import ch.alpine.subare.core.util.FeatureWeight;
import ch.alpine.subare.core.util.Infoline;
import ch.alpine.subare.core.util.LearningRate;
import ch.alpine.subare.core.util.PolicyType;
import ch.alpine.subare.core.util.gfx.StateActionRasters;
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
        new GifAnimationWriter(HomeDirectory.Pictures("gambler_tos_" + name + ".gif"), 250, TimeUnit.MILLISECONDS)) {
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

  public static void main(String[] args) throws Exception {
    run(SarsaType.ORIGINAL);
    run(SarsaType.EXPECTED);
    run(SarsaType.QLEARNING);
  }
}
