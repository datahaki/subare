// code by jph
package ch.alpine.subare.ch05.racetrack;

import java.io.File;
import java.io.IOException;

import ch.alpine.subare.core.Policy;
import ch.alpine.subare.core.alg.IterativePolicyEvaluation;
import ch.alpine.subare.core.util.DiscreteUtils;
import ch.alpine.subare.core.util.EquiprobablePolicy;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.io.Import;
import ch.alpine.tensor.sca.Round;

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
