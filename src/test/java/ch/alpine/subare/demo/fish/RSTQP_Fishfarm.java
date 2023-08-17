// code by jph
package ch.alpine.subare.demo.fish;

import java.util.concurrent.TimeUnit;

import ch.alpine.subare.alg.Random1StepTabularQPlanning;
import ch.alpine.subare.util.ConstantLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.TabularSteps;
import ch.alpine.subare.util.gfx.StateRasters;
import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;
import ch.alpine.tensor.sca.Round;

enum RSTQP_Fishfarm {
  ;
  public static void main(String[] args) throws Exception {
    Fishfarm fishfarm = new Fishfarm(20, 20);
    FishfarmRaster cliffwalkRaster = new FishfarmRaster(fishfarm);
    final DiscreteQsa ref = FishfarmHelper.getOptimalQsa(fishfarm);
    DiscreteQsa qsa = DiscreteQsa.build(fishfarm, DoubleScalar.POSITIVE_INFINITY);
    Random1StepTabularQPlanning rstqp = Random1StepTabularQPlanning.of( //
        fishfarm, qsa, ConstantLearningRate.one());
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures("fishfarm_qsa_rstqp.gif"), 200, TimeUnit.MILLISECONDS)) {
      int batches = 20;
      for (int index = 0; index < batches; ++index) {
        Infoline infoline = Infoline.print(fishfarm, index, ref, qsa);
        TabularSteps.batch(fishfarm, fishfarm, rstqp);
        animationWriter.write(StateRasters.qsaLossRef(cliffwalkRaster, qsa, ref));
        if (infoline.isLossfree())
          break;
      }
    }
    DiscreteUtils.print(qsa, Round._2);
  }
}
