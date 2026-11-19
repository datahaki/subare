// code by jph
package ch.alpine.subare.net.reg;

import ch.alpine.subare.net.api.NetRegularization;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

public enum NoRegularization implements NetRegularization {
  INSTANCE;

  @Override
  public Scalar penalty(boolean decayInLoss, Tensor weights) {
    return RealScalar.ZERO;
  }

  @Override
  public Tensor gradient(boolean decayInLoss, Tensor weights) {
    return weights.maps(Scalar::zero);
  }

  @Override
  public Tensor sneakDecay(boolean decayInLoss, Tensor weights) {
    return weights.maps(Scalar::zero);
  }

  @Override
  public String toString() {
    return "NoReg";
  }
}
