// code by jph
package ch.ethz.idsc.subare.core.alg;

import ch.alpine.tensor.Scalar;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;

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
