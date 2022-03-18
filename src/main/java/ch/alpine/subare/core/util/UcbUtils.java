// code by jph, fluric
package ch.alpine.subare.core.util;

import ch.alpine.subare.core.DiscreteModel;
import ch.alpine.subare.core.QsaInterface;
import ch.alpine.subare.core.StateActionCounter;
import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.sca.Sign;
import ch.alpine.tensor.sca.pow.Sqrt;

public enum UcbUtils {
  ;
  public static DiscreteQsa getUcbInQsa(DiscreteModel discreteModel, QsaInterface qsa, StateActionCounter sac) {
    DiscreteQsa qsaWithUcb = DiscreteQsa.build(discreteModel);
    for (Tensor state : discreteModel.states())
      for (Tensor action : discreteModel.actions(state))
        qsaWithUcb.assign(state, action, getUpperConfidenceBound(state, action, qsa.value(state, action), sac));
    return qsaWithUcb;
  }

  public static Scalar getUpperConfidenceBound(Tensor state, Tensor action, Scalar qsaValue, StateActionCounter sac) {
    Tensor key = StateAction.key(state, action);
    Scalar Nta = sac.stateActionCount(key);
    if (Scalars.isZero(Nta))
      return DoubleScalar.POSITIVE_INFINITY;
    Scalar bias = Sqrt.of(sac.stateCount(state)).divide(Nta);
    Scalar sign = Sign.isPositive(qsaValue) //
        ? RealScalar.ONE
        : RealScalar.of(-1);
    return qsaValue.multiply((RealScalar.ONE.add(bias.multiply(sign))));
  }
}
