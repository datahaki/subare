// code by jph
package ch.alpine.subare.book.ch06.walk;

import ch.alpine.subare.api.Policy;
import ch.alpine.subare.mc.FirstVisitPolicyEvaluation;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.DiscreteVs;
import ch.alpine.subare.util.EquiprobablePolicy;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.tensor.sca.Round;

/** first visit policy evaluation determines state values v(s)
 * 
 * <pre>
 * 0 0
 * 1 0.16
 * 2 0.30
 * 3 0.47
 * 4 0.64
 * 5 0.84
 * 6 0
 * </pre> */
enum FVPE_Randomwalk {
  ;
  static void main() {
    Randomwalk randomwalk = new Randomwalk(5);
    FirstVisitPolicyEvaluation fvpe = new FirstVisitPolicyEvaluation( //
        randomwalk, null);
    Policy policy = EquiprobablePolicy.create(randomwalk);
    for (int count = 0; count < 100; ++count)
      ExploringStarts.batch(randomwalk, policy, fvpe);
    DiscreteVs vs = fvpe.vs();
    DiscreteUtils.print(vs, Round._2);
  }
}
