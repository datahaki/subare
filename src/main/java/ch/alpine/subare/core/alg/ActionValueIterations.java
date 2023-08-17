// code by jph
package ch.alpine.subare.core.alg;

import ch.alpine.subare.core.api.StandardModel;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.tensor.Scalar;

/**  */
public enum ActionValueIterations {
  ;
  /** @param standardModel
   * @param threshold positive
   * @return */
  public static DiscreteQsa solve(StandardModel standardModel, Scalar threshold) {
    ActionValueIteration actionValueIteration = ActionValueIteration.of(standardModel);
    actionValueIteration.untilBelow(threshold);
    return actionValueIteration.qsa();
  }
}
