// code by jph
package ch.alpine.subare.ch06.windy;

import java.util.concurrent.TimeUnit;

import ch.alpine.subare.core.alg.Random1StepTabularQPlanning;
import ch.alpine.subare.core.util.ConstantLearningRate;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.subare.core.util.Infoline;
import ch.alpine.subare.core.util.TabularSteps;
import ch.alpine.subare.core.util.gfx.StateActionRasters;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;

/** the R1STQP algorithm is cheating on the Windygrid
 * because TabularSteps starts in every state-action pair
 * instead of only the 1 start state of Windygrid */
enum RSTQP_Windygrid {
  ;
  public static void main(String[] args) throws Exception {
    Windygrid windygrid = Windygrid.createFour();
    WindygridRaster windygridRaster = new WindygridRaster(windygrid);
    final DiscreteQsa ref = WindygridHelper.getOptimalQsa(windygrid);
    DiscreteQsa qsa = DiscreteQsa.build(windygrid);
    Random1StepTabularQPlanning rstqp = Random1StepTabularQPlanning.of( //
        windygrid, qsa, ConstantLearningRate.one());
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures("windygrid_qsa_rstqp.gif"), 250, TimeUnit.MILLISECONDS)) {
      int batches = 40;
      for (int index = 0; index < batches; ++index) {
        Infoline infoline = Infoline.print(windygrid, index, ref, qsa);
        TabularSteps.batch(windygrid, windygrid, rstqp);
        animationWriter.write(StateActionRasters.qsaLossRef(windygridRaster, qsa, ref));
        if (infoline.isLossfree())
          break;
      }
    }
  }
}
