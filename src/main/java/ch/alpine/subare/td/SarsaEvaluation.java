// code by jph and fluric
package ch.alpine.subare.td;

import ch.alpine.subare.util.PolicyExt;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

/* package */ interface SarsaEvaluation {
  Scalar evaluate(Tensor state, PolicyExt policy);

  /** The action probabilities are chosen according to policy1 and then added up by the qsa of policy2
   * 
   * @param state
   * @param policy1
   * @param policy2
   * @return qsa value */
  Scalar crossEvaluate(Tensor state, PolicyExt policy1, PolicyExt policy2);
}
