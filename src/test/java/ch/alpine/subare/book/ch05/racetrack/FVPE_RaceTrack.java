// code by jph
package ch.alpine.subare.book.ch05.racetrack;

import java.io.File;
import java.io.IOException;

import ch.alpine.subare.api.Policy;
import ch.alpine.subare.mc.FirstVisitPolicyEvaluation;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.DiscreteVs;
import ch.alpine.subare.util.EquiprobablePolicy;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.tensor.Unprotect;
import ch.alpine.tensor.io.Import;
import ch.alpine.tensor.sca.Round;

enum FVPE_RaceTrack {
  ;
  public static void main(String[] args) throws IOException {
    File file = Unprotect.file("/ch05/track0.png");
    Racetrack racetrack = new Racetrack(Import.of(file), 3);
    FirstVisitPolicyEvaluation fvpe = new FirstVisitPolicyEvaluation( //
        racetrack, null);
    Policy policy = EquiprobablePolicy.create(racetrack);
    for (int count = 0; count < 10; ++count)
      ExploringStarts.batch(racetrack, policy, fvpe);
    DiscreteVs vs = fvpe.vs();
    DiscreteUtils.print(vs, Round._1);
  }
}
