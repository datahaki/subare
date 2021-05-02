// code by fluric
package ch.ethz.idsc.subare.analysis;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.nrm.Vector2Norm;
import ch.alpine.tensor.nrm.Vector2NormSquared;
import ch.alpine.tensor.red.Total;
import ch.alpine.tensor.sca.Power;
import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteValueFunctions;
import ch.ethz.idsc.subare.core.util.Loss;

public enum DiscreteModelErrorAnalysis {
  LINEAR_QSA() {
    @Override
    public Scalar getError(DiscreteModel discreteModel, DiscreteQsa refQsa, DiscreteQsa currentQsa) {
      return DiscreteValueFunctions.distance(refQsa, currentQsa);
    }
  },
  SQUARE_QSA() {
    @Override
    public Scalar getError(DiscreteModel discreteModel, DiscreteQsa refQsa, DiscreteQsa currentQsa) {
      return Power.of(DiscreteValueFunctions.distance(refQsa, currentQsa, Vector2Norm::of), 2);
    }
  },
  LINEAR_POLICY() {
    @Override
    public Scalar getError(DiscreteModel discreteModel, DiscreteQsa refQsa, DiscreteQsa currentQsa) {
      return Total.ofVector(Loss.asQsa(discreteModel, refQsa, currentQsa).values());
    }
  },
  SQUARE_POLICY() {
    @Override
    public Scalar getError(DiscreteModel discreteModel, DiscreteQsa refQsa, DiscreteQsa currentQsa) {
      return Vector2NormSquared.of(Loss.asQsa(discreteModel, refQsa, currentQsa).values());
    }
  };

  /** @param discreteModel
   * @param refQsa
   * @param currentQsa
   * @return */
  public abstract Scalar getError(DiscreteModel discreteModel, DiscreteQsa refQsa, DiscreteQsa currentQsa);
}
