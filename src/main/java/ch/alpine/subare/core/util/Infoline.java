// code by jph
package ch.alpine.subare.core.util;

import ch.alpine.subare.core.api.DiscreteModel;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.sca.Chop;
import ch.alpine.tensor.sca.Round;

public class Infoline {
  public static Infoline print(DiscreteModel discreteModel, int index, DiscreteQsa ref, DiscreteQsa qsa) {
    Infoline infoline = new Infoline(discreteModel, ref, qsa);
    System.out.printf("%2d %8s  %s%n", index, infoline.error.map(Round._1), infoline.loss);
    return infoline;
  }

  // ---
  private final Scalar error;
  private final Scalar loss;

  public Infoline(DiscreteModel discreteModel, DiscreteQsa ref, DiscreteQsa qsa) {
    this.error = DiscreteValueFunctions.distance(qsa, ref);
    this.loss = Loss.accumulation(discreteModel, DiscreteValueFunctions.numeric(ref), qsa);
  }

  public Scalar q_error() {
    return error;
  }

  public Scalar loss() {
    return loss;
  }

  public boolean isLossfree() {
    return Chop._10.isZero(loss);
  }

  public boolean isErrorFree() {
    return Chop._10.isZero(error);
  }
}
