// code by jph
package ch.alpine.subare.demo.bus;

import java.io.IOException;

import ch.alpine.subare.core.alg.ActionValueIterations;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.subare.core.util.gfx.StateActionRasters;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.Export;

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
