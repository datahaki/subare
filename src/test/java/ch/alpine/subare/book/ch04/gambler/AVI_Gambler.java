// code by jph
package ch.alpine.subare.book.ch04.gambler;

import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.DiscreteVs;
import ch.alpine.subare.util.gfx.StateActionRasters;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.Export;
import ch.alpine.tensor.io.Put;

/** action value iteration for gambler's dilemma
 * 
 * visualizes the exact optimal policy */
/* package */ enum AVI_Gambler {
  ;
  static void main() throws Exception {
    GamblerModel gamblerModel = new GamblerModel(100, RealScalar.of(0.35));
    GamblerRaster gamblerRaster = new GamblerRaster(gamblerModel) {
      @Override
      public int magnify() {
        return 1;
      }
    };
    DiscreteQsa ref = GamblerHelper.getOptimalQsa(gamblerModel);
    Export.of(HomeDirectory.Pictures("gambler_qsa.png"), //
        StateActionRasters.qsa(gamblerRaster, ref));
    Export.of(HomeDirectory.Pictures("gambler_qsa_avi.png"), //
        StateActionRasters.qsaPolicy(gamblerRaster, ref));
    DiscreteVs vs = DiscreteUtils.createVs(gamblerModel, ref);
    Put.of(HomeDirectory.file("ex403_vs_values"), vs.values());
    System.out.println("done.");
  }
}
