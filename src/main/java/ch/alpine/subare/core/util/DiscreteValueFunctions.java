// code by jph
package ch.alpine.subare.core.util;

import ch.alpine.subare.core.api.StateActionCounter;
import ch.alpine.tensor.RationalScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Rescale;
import ch.alpine.tensor.api.TensorScalarFunction;
import ch.alpine.tensor.nrm.Vector1Norm;
import ch.alpine.tensor.red.Times;
import ch.alpine.tensor.sca.InvertUnlessZero;
import ch.alpine.tensor.sca.N;
import ch.alpine.tensor.sca.exp.LogisticSigmoid;

public enum DiscreteValueFunctions {
  ;
  @SuppressWarnings("unchecked")
  public static <T extends DiscreteValueFunction> T numeric(T tvi) {
    return (T) tvi.create(tvi.values().map(N.DOUBLE).stream());
  }

  @SuppressWarnings("unchecked")
  public static <T extends DiscreteValueFunction> T rescaled(T tvi) {
    return (T) tvi.create(Rescale.of(tvi.values()).stream());
  }

  /** @param tvi1
   * @param tvi2
   * @param norm for vectors
   * @return */
  public static Scalar distance(DiscreteValueFunction tvi1, DiscreteValueFunction tvi2, TensorScalarFunction norm) {
    return norm.apply(_difference(tvi1, tvi2));
  }

  public static Scalar distance(DiscreteValueFunction tvi1, DiscreteValueFunction tvi2) {
    return distance(tvi1, tvi2, Vector1Norm::of);
  }

  @SuppressWarnings("unchecked")
  public static <T extends DiscreteValueFunction> T average(T tvi1, T tvi2) {
    return (T) tvi1.create(tvi1.values().add(tvi2.values()).multiply(RationalScalar.HALF).stream());
  }

  @SuppressWarnings("unchecked")
  public static <T extends DiscreteValueFunction> T logisticDifference(T tvi1, T tvi2) {
    return (T) tvi1.create(_difference(tvi1, tvi2).map(LogisticSigmoid.FUNCTION).stream());
  }

  @SuppressWarnings("unchecked")
  public static <T extends DiscreteValueFunction> T logisticDifference(T tvi1, T tvi2, Scalar factor) {
    return (T) tvi1.create(_difference(tvi1, tvi2).multiply(factor).map(LogisticSigmoid.FUNCTION).stream());
  }

  /** @param qsa1
   * @param qsa2
   * @param sac1
   * @param sac2
   * @return the weighted average of the qsa values according to the number of visits occurred in the different {@link LearningRate}'s.
   * For each element of the qsa: qsa(e) = (qsa1(e)*lr1_visits(e) + qsa2(e)*lr2_visits(e))/(lr1_visits(e)+lr2_visits(e)) */
  public static DiscreteQsa weightedAverage(DiscreteQsa qsa1, DiscreteQsa qsa2, //
      StateActionCounter sac1, StateActionCounter sac2) {
    Tensor visits1 = Tensor.of(qsa1.keys().stream().map(sac1::stateActionCount));
    Tensor visits2 = Tensor.of(qsa2.keys().stream().map(sac2::stateActionCount));
    Tensor inverse = visits1.add(visits2).map(InvertUnlessZero.FUNCTION);
    return qsa1.create( //
        Times.of(Times.of(qsa1.values(), visits1).add(Times.of(qsa2.values(), visits2)), inverse).stream());
  }

  // ---
  // helper function
  private static boolean _isCompatible(DiscreteValueFunction tvi1, DiscreteValueFunction tvi2) {
    return tvi1.keys().equals(tvi2.keys());
  }

  // helper function
  private static Tensor _difference(DiscreteValueFunction tvi1, DiscreteValueFunction tvi2) {
    if (_isCompatible(tvi1, tvi2))
      return tvi1.values().subtract(tvi2.values());
    throw new IllegalArgumentException();
  }
}
