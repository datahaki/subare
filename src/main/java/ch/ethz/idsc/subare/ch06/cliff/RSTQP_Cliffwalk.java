// code by jph
package ch.ethz.idsc.subare.ch06.cliff;

import java.util.concurrent.TimeUnit;

import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;
import ch.alpine.tensor.sca.Round;
import ch.ethz.idsc.subare.core.alg.Random1StepTabularQPlanning;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.TabularSteps;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;

enum RSTQP_Cliffwalk {
  ;
  public static void main(String[] args) throws Exception {
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    CliffwalkRaster cliffwalkRaster = new CliffwalkRaster(cliffwalk);
    final DiscreteQsa ref = CliffwalkHelper.getOptimalQsa(cliffwalk);
    DiscreteQsa qsa = DiscreteQsa.build(cliffwalk, DoubleScalar.POSITIVE_INFINITY);
    Random1StepTabularQPlanning rstqp = Random1StepTabularQPlanning.of( //
        cliffwalk, qsa, ConstantLearningRate.one());
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures("cliffwalk_qsa_rstqp.gif"), 200, TimeUnit.MILLISECONDS)) {
      int batches = 20;
      for (int index = 0; index < batches; ++index) {
        Infoline infoline = Infoline.print(cliffwalk, index, ref, qsa);
        TabularSteps.batch(cliffwalk, cliffwalk, rstqp);
        animationWriter.write(StateActionRasters.qsaLossRef(cliffwalkRaster, qsa, ref));
        if (infoline.isLossfree())
          break;
      }
    }
    DiscreteUtils.print(qsa, Round._2);
  }
}
