// code by jph
package ch.ethz.idsc.subare.demo.bus;

import java.io.IOException;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.Export;
import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;

/* package */ enum AVI_Charger {
  ;
  public static void main(String[] args) throws IOException {
    TripProfile tripProfile = new ConstantDrawTrip(24, 3);
    Charger charger = new Charger(tripProfile, 6);
    DiscreteQsa ref = ActionValueIterations.solve(charger, RealScalar.of(.0001));
    ChargerRaster chargerRaster = new ChargerRaster(charger);
    Export.of(HomeDirectory.Pictures("charger_qsa_avi.png"), //
        StateActionRasters.qsaPolicy(chargerRaster, ref));
    // ref.print();
  }
}
