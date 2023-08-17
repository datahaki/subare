// code by jph
package ch.alpine.subare.ch04.gambler;

import ch.alpine.subare.core.api.Policy;
import ch.alpine.subare.core.mc.FirstVisitPolicyEvaluation;
import ch.alpine.subare.core.util.DiscreteValueFunctions;
import ch.alpine.subare.core.util.DiscreteVs;
import ch.alpine.subare.core.util.ExploringStarts;
import ch.alpine.subare.core.util.PolicyType;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.sca.N;

/** FirstVisitPolicyEvaluation of optimal greedy policy */
/* package */ enum FVPE_Gambler {
  ;
  public static void main(String[] args) {
    GamblerModel gamblerModel = GamblerModel.createDefault();
    DiscreteVs ref = GamblerHelper.getOptimalVs(gamblerModel);
    Policy policy = PolicyType.GREEDY.bestEquiprobable(gamblerModel, ref, null);
    FirstVisitPolicyEvaluation fvpe = new FirstVisitPolicyEvaluation( //
        gamblerModel, null);
    for (int count = 0; count < 100; ++count) {
      ExploringStarts.batch(gamblerModel, policy, fvpe);
      DiscreteVs vs = fvpe.vs();
      Scalar diff = DiscreteValueFunctions.distance(vs, ref);
      System.out.println(count + " " + N.DOUBLE.apply(diff));
    }
  }
}
