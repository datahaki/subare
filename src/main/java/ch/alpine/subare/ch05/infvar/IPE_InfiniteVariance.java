// code by jph
package ch.alpine.subare.ch05.infvar;

import ch.alpine.subare.core.Policy;
import ch.alpine.subare.core.StandardModel;
import ch.alpine.subare.core.alg.IterativePolicyEvaluation;
import ch.alpine.subare.core.util.DiscreteUtils;
import ch.alpine.tensor.RationalScalar;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.sca.Round;

// TODO check again
enum IPE_InfiniteVariance {
  ;
  public static void main(String[] args) {
    StandardModel standardModel = new InfiniteVariance();
    Policy policy = new ConstantPolicy(RationalScalar.of(9, 10));
    IterativePolicyEvaluation a = new IterativePolicyEvaluation( //
        standardModel, policy);
    a.until(RealScalar.of(.0001));
    System.out.println(a.iterations());
    DiscreteUtils.print(a.vs(), Round._2);
  }
}