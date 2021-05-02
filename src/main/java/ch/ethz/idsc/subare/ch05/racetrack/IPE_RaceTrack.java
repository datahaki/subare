// code by jph
package ch.ethz.idsc.subare.ch05.racetrack;

import java.io.File;
import java.io.IOException;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.io.Import;
import ch.alpine.tensor.sca.Round;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.alg.IterativePolicyEvaluation;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;

enum IPE_RaceTrack {
  ;
  public static void main(String[] args) throws IOException {
    String path = "".getClass().getResource("/ch05/track0.png").getPath();
    Racetrack racetrack = new Racetrack(Import.of(new File(path)), 3);
    Policy policy = EquiprobablePolicy.create(racetrack);
    IterativePolicyEvaluation ipe = new IterativePolicyEvaluation(racetrack, policy);
    ipe.until(RealScalar.of(.1));
    DiscreteUtils.print(ipe.vs(), Round._1);
  }
}
