// code by jph
package ch.alpine.subare.util;

import ch.alpine.subare.api.mod.DiscreteModel;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.sca.Chop;
import ch.alpine.tensor.sca.Round;

public record Infoline(Scalar error, Scalar loss) {
  public static Infoline of(DiscreteModel discreteModel, DiscreteQsa ref, DiscreteQsa qsa) {
    return new Infoline( //
        DiscreteValueFunctions.distance(qsa, ref), //
        Loss.accumulation(discreteModel, DiscreteValueFunctions.numeric(ref), qsa));
  }

  public boolean isLossfree() {
    return Chop._10.isZero(loss);
  }

  public boolean isErrorFree() {
    return Chop._10.isZero(error);
  }

  public Tensor vector() {
    return Tensors.of(error, loss);
  }

  public Tensor indexedVector(int index) {
    return Tensors.of(RealScalar.of(index), error, loss);
  }

  @Override
  public final String toString() {
    return error.maps(Round._3) + " " + loss.maps(Round._3);
  }
}
