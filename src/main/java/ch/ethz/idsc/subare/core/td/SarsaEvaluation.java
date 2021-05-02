// code by jph and fluric
package ch.ethz.idsc.subare.core.td;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.ethz.idsc.subare.core.util.PolicyExt;

/* package */ interface SarsaEvaluation {
  Scalar evaluate(Tensor state, PolicyExt policy);

  /** The action probabilities are chosen according to policy1 and then added up by the qsa of policy2
   * @param state
   * @param policy1
   * @param policy2
   * @return qsa value */
  Scalar crossEvaluate(Tensor state, PolicyExt policy1, PolicyExt policy2);
}
