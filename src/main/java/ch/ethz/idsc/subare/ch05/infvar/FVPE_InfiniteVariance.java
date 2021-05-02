// code by jph
package ch.ethz.idsc.subare.ch05.infvar;

import ch.alpine.tensor.RationalScalar;
import ch.alpine.tensor.sca.N;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.mc.FirstVisitPolicyEvaluation;
import ch.ethz.idsc.subare.core.util.ExploringStarts;

/* package */ enum FVPE_InfiniteVariance {
  ;
  public static void main(String[] args) {
    InfiniteVariance infiniteVariance = new InfiniteVariance();
    FirstVisitPolicyEvaluation fvpe = new FirstVisitPolicyEvaluation( //
        infiniteVariance, null);
    Policy policy = new ConstantPolicy(RationalScalar.of(5, 10));
    for (int count = 0; count < 100; ++count)
      ExploringStarts.batch(infiniteVariance, policy, fvpe);
    System.out.println(N.DOUBLE.of(fvpe.vs().values()));
  }
}
