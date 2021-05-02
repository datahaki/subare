// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import java.util.concurrent.TimeUnit;

import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;
import ch.alpine.tensor.io.ImageFormat;
import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;

/** action value iteration for gambler's dilemma
 * 
 * visualizes each pass of the action value iteration */
/* package */ enum AVI_GamblerAnimation {
  ;
  public static void main(String[] args) throws Exception {
    GamblerModel gamblerModel = GamblerModel.createDefault();
    final DiscreteQsa ref = GamblerHelper.getOptimalQsa(gamblerModel);
    ActionValueIteration avi = ActionValueIteration.of(gamblerModel);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures("gambler_qsa_avi.gif"), 500, TimeUnit.MILLISECONDS)) {
      for (int index = 0; index < 13; ++index) {
        DiscreteQsa qsa = avi.qsa();
        Infoline.print(gamblerModel, index, ref, qsa);
        animationWriter.write(StateActionRasters.qsaPolicyRef(new GamblerRaster(gamblerModel), qsa, ref));
        avi.step();
      }
      animationWriter.write(ImageFormat.of(StateActionRasters.qsaPolicyRef(new GamblerRaster(gamblerModel), avi.qsa(), ref)));
    }
  }
}
