// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch08.maze;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.td.PrioritizedSweeping;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.StepExploringStarts;
import ch.ethz.idsc.subare.core.util.gfx.StateRasters;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

/** determines q(s,a) function for equiprobable "random" policy */
class PS_Dynamaze {
  static void handle(SarsaType sarsaType, int batches) throws Exception {
    System.out.println(sarsaType);
    String name = "maze2";
    Dynamaze dynamaze;
    // dynamaze = DynamazeHelper.original(name);
    dynamaze = DynamazeHelper.create5(3);
    DynamazeRaster dynamazeRaster = new DynamazeRaster(dynamaze);
    final DiscreteQsa ref = DynamazeHelper.getOptimalQsa(dynamaze);
    DiscreteQsa qsa = DiscreteQsa.build(dynamaze);
    Tensor epsilon = Subdivide.of(.3, .01, batches);
    LearningRate learningRate = DefaultLearningRate.of(5, 1.01);
    Sarsa sarsa = sarsaType.supply(dynamaze, qsa, learningRate);
    PrioritizedSweeping prioritizedSweeping = new PrioritizedSweeping( //
        sarsa, 10, RealScalar.ZERO);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.Pictures(name + "_ps_" + sarsaType + ".gif"), 250);
    // ---
    StepExploringStarts stepExploringStarts = //
        new StepExploringStarts(dynamaze, prioritizedSweeping) {
          @Override
          public Policy batchPolicy(int batch) {
            Policy policy = EGreedyPolicy.bestEquiprobable(dynamaze, qsa, epsilon.Get(batch));
            // prioritizedSweeping.setPolicy(policy);
            return policy;
          }
        };
    while (stepExploringStarts.batchIndex() < batches) {
      Infoline infoline = Infoline.print(dynamaze, stepExploringStarts.batchIndex(), ref, qsa);
      sarsa.setExplore(epsilon.Get(stepExploringStarts.batchIndex()));
      stepExploringStarts.nextEpisode();
      gsw.append(ImageFormat.of(StateRasters.qsaLossRef(dynamazeRaster, qsa, ref)));
      if (infoline.isLossfree())
        break;
    }
    gsw.close();
  }

  public static void main(String[] args) throws Exception {
    // handle(SarsaType.original, 10);
    // handle(SarsaType.expected, 50);
    handle(SarsaType.qlearning, 10);
  }
}
