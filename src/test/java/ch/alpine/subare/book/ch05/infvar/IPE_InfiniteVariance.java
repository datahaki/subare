// code by jph
package ch.alpine.subare.book.ch05.infvar;

import ch.alpine.subare.alg.IterativePolicyEvaluation;
import ch.alpine.subare.api.Policy;
import ch.alpine.subare.api.StandardModel;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.tensor.RationalScalar;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.sca.Round;

// TODO SUBARE check again
enum IPE_InfiniteVariance {
  ;
  static void main() {
    StandardModel standardModel = new InfiniteVariance();
    Policy policy = new ConstantPolicy(RationalScalar.of(9, 10));
    IterativePolicyEvaluation a = new IterativePolicyEvaluation( //
        standardModel, policy);
    a.until(RealScalar.of(.0001));
    System.out.println(a.iterations());
    DiscreteUtils.print(a.vs(), Round._2);
  }
}
