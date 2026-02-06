// code by jph
package ch.alpine.subare.book.ch05.wireloop;

import java.util.concurrent.TimeUnit;

import ch.alpine.subare.api.FeatureMapper;
import ch.alpine.subare.api.LearningRate;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.td.TrueOnlineSarsa;
import ch.alpine.subare.util.DefaultLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.ExactFeatureMapper;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.FeatureWeight;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.PolicyBase;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.subare.util.gfx.StateRasters;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.ext.Timing;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;

enum TOS_Wireloop {
  ;
  private static final Scalar LAMBDA = RealScalar.of(0.3);

  static void run(SarsaType sarsaType) throws Exception {
    String name = "wire4";
    System.out.println(sarsaType);
    WireloopReward wireloopReward = WireloopReward.freeSteps();
    wireloopReward = WireloopReward.constantCost();
    Wireloop wireloop = WireloopHelper.create(name, WireloopReward::id_x, wireloopReward);
    // Gambler gambler = new Gambler(20, RealScalar.of(.4));
    final DiscreteQsa ref = WireloopHelper.getOptimalQsa(wireloop);
    FeatureMapper mapper = ExactFeatureMapper.of(wireloop);
    FeatureWeight w = new FeatureWeight(mapper);
    // Tensor epsilon = Subdivide.of(.2, .01, batches); // used in egreedy
    // DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    LearningRate learningRate = DefaultLearningRate.of(RealScalar.of(3), RealScalar.of(0.81));
    StateActionCounter sac = new DiscreteStateActionCounter();
    PolicyBase policy = PolicyType.EGREEDY.bestEquiprobable(wireloop, DiscreteQsa.build(wireloop), sac);
    // LearningRate learningRate = ConstantLearningRate.of(RealScalar.of(0.3), false); // the case without warmStart
    TrueOnlineSarsa trueOnlineSarsa = sarsaType.trueOnline(wireloop, LAMBDA, mapper, learningRate, w, sac, policy);
    final String algo = sarsaType.name().toLowerCase();
    Timing timing = Timing.started();
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures.resolve(name + "_tos_" + algo + ".gif"), 250, TimeUnit.MILLISECONDS)) {
      for (int batch = 0; batch < 20; ++batch) {
        // System.out.println("batch " + batch);
        policy.setQsa(trueOnlineSarsa.qsaInterface());
        ExploringStarts.batch(wireloop, policy, trueOnlineSarsa);
        // DiscreteQsa toQsa = trueOnlineSarsa.getQsa();
        // XYtoSarsa.append(Tensors.vector(RealScalar.of(index).number(), errorAnalysis.getError(monteCarloInterface, optimalQsa, toQsa).number()));
        DiscreteQsa qsa = trueOnlineSarsa.qsa();
        Infoline infoline = Infoline.print(wireloop, batch, ref, qsa);
        animationWriter.write(StateRasters.qsaLossRef(new WireloopRaster(wireloop), qsa, ref));
        if (infoline.isLossfree()) {
          animationWriter.write(StateRasters.qsaLossRef(new WireloopRaster(wireloop), qsa, ref));
          animationWriter.write(StateRasters.qsaLossRef(new WireloopRaster(wireloop), qsa, ref));
          animationWriter.write(StateRasters.qsaLossRef(new WireloopRaster(wireloop), qsa, ref));
          break;
        }
      }
    }
    System.out.println("Time for TrueOnlineSarsa: " + timing.seconds() + "s");
  }

  static void main() throws Exception {
    for (SarsaType sarsaType : SarsaType.values())
      run(sarsaType);
  }
}
