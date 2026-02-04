// code by jph
package ch.alpine.subare.book.ch04.grid;

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
import ch.alpine.subare.util.gfx.StateActionRasters;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.ext.Timing;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;

enum TOS_Gridworld {
  ;
  private static final Scalar LAMBDA = RealScalar.of(0.5);

  static void run(SarsaType sarsaType) throws Exception {
    System.out.println(sarsaType);
    Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    FeatureMapper mapper = ExactFeatureMapper.of(gridworld);
    FeatureWeight w = new FeatureWeight(mapper);
    // Tensor epsilon = Subdivide.of(.2, .01, batches); // used in egreedy
    // DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    LearningRate learningRate = DefaultLearningRate.of(RealScalar.of(3), RealScalar.of(0.81));
    StateActionCounter sac = new DiscreteStateActionCounter();
    PolicyBase policy = PolicyType.EGREEDY.bestEquiprobable(gridworld, DiscreteQsa.build(gridworld), sac);
    // LearningRate learningRate = ConstantLearningRate.of(RealScalar.of(0.3), false); // the case without warmStart
    TrueOnlineSarsa trueOnlineSarsa = sarsaType.trueOnline(gridworld, LAMBDA, mapper, learningRate, w, sac, policy);
    final String name = sarsaType.name().toLowerCase();
    Timing timing = Timing.started();
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures("gridworld_tos_" + name + ".gif"), 250, TimeUnit.MILLISECONDS)) {
      for (int batch = 0; batch < 100; ++batch) {
        // System.out.println("starting batch " + (index + 1) + " of " + batches);
        policy.setQsa(trueOnlineSarsa.qsaInterface());
        ExploringStarts.batch(gridworld, policy, trueOnlineSarsa);
        // DiscreteQsa toQsa = trueOnlineSarsa.getQsa();
        // XYtoSarsa.append(Tensors.vector(RealScalar.of(index).number(), errorAnalysis.getError(monteCarloInterface, optimalQsa, toQsa).number()));
        DiscreteQsa qsa = trueOnlineSarsa.qsa();
        Infoline infoline = Infoline.print(gridworld, batch, ref, qsa);
        animationWriter.write(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), qsa, ref));
        if (infoline.isLossfree()) {
          animationWriter.write(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), qsa, ref));
          animationWriter.write(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), qsa, ref));
          animationWriter.write(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), qsa, ref));
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
