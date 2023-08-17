// code by jph
package ch.alpine.subare.core.alg;

import ch.alpine.subare.core.api.StandardModel;
import ch.alpine.subare.core.util.DiscreteVs;
import ch.alpine.tensor.Scalar;

public enum ValueIterations {
  ;
  public static DiscreteVs solve(StandardModel standardModel, Scalar threshold) {
    ValueIteration valueIteration = new ValueIteration(standardModel);
    valueIteration.untilBelow(threshold);
    return valueIteration.vs();
  }
}
