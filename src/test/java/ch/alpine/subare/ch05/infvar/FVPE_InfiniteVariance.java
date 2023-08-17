// code by jph
package ch.alpine.subare.ch05.infvar;

import ch.alpine.subare.core.api.Policy;
import ch.alpine.subare.core.mc.FirstVisitPolicyEvaluation;
import ch.alpine.subare.core.util.ExploringStarts;
import ch.alpine.tensor.RationalScalar;
import ch.alpine.tensor.sca.N;

/* package */ enum FVPE_InfiniteVariance {
  ;
  public static void main(String[] args) {
    InfiniteVariance infiniteVariance = new InfiniteVariance();
    FirstVisitPolicyEvaluation fvpe = new FirstVisitPolicyEvaluation( //
        infiniteVariance, null);
    Policy policy = new ConstantPolicy(RationalScalar.of(5, 10));
    for (int count = 0; count < 100; ++count)
      ExploringStarts.batch(infiniteVariance, policy, fvpe);
    System.out.println(fvpe.vs().values().map(N.DOUBLE));
  }
}
