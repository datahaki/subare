// code by jph
package ch.alpine.subare.book.ch04.gambler;

import java.util.concurrent.TimeUnit;

import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.gfx.StateActionRasters;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.AnimationWriter;
import ch.alpine.tensor.io.GifAnimationWriter;
import ch.alpine.tensor.io.ImageFormat;

/** action value iteration for gambler's dilemma
 * 
 * visualizes each pass of the action value iteration */
/* package */ enum AVI_GamblerAnimation {
  ;
  static void main() throws Exception {
    GamblerModel gamblerModel = GamblerModel.createDefault();
    final DiscreteQsa ref = GamblerHelper.getOptimalQsa(gamblerModel);
    ActionValueIteration avi = ActionValueIteration.of(gamblerModel);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures.resolve("gambler_qsa_avi.gif"), 500, TimeUnit.MILLISECONDS)) {
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
