// code by jph
package ch.alpine.subare.ch04.grid;

import java.util.concurrent.TimeUnit;

import ch.alpine.subare.core.EpisodeVsEstimator;
import ch.alpine.subare.core.Policy;
import ch.alpine.subare.core.mc.ConstantAlphaMonteCarloVs;
import ch.alpine.subare.core.util.DefaultLearningRate;
import ch.alpine.subare.core.util.DiscreteValueFunctions;
import ch.alpine.subare.core.util.EquiprobablePolicy;
import ch.alpine.subare.core.util.ExploringStarts;
import ch.alpine.subare.core.util.gfx.StateRasters;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;

/* package */ enum CAMC_Gridworld { // LONGTERM work in progress?
  ;
  public static void main(String[] args) throws Exception {
    Gridworld gridworld = new Gridworld();
    GridworldRaster gridworldRaster = new GridworldRaster(gridworld);
    // final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    EpisodeVsEstimator camc = ConstantAlphaMonteCarloVs.create( //
        gridworld, DefaultLearningRate.of(3, .51));
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures("gridworld_vs_camc.gif"), 100, TimeUnit.MILLISECONDS)) {
      final int batches = 50;
      // Tensor epsilon = Subdivide.of(.2, .05, batches);
      for (int index = 0; index < batches; ++index) {
        System.out.println(index);
        for (int count = 0; count < 20; ++count) {
          Policy policy = EquiprobablePolicy.create(gridworld);
          // EGreedyPolicy.bestEquiprobable(gridworld, camc.vs(), epsilon.Get(index));
          ExploringStarts.batch(gridworld, policy, camc);
        }
        animationWriter.write(StateRasters.vs(gridworldRaster, DiscreteValueFunctions.rescaled(camc.vs())));
      }
    }
  }
}
