package ch.alpine.subare.net.reg;

import ch.alpine.subare.net.api.NetRegularization;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

public enum NoWeightDecay implements NetRegularization {
  INSTANCE;

  @Override
  public Scalar penalty(boolean weightDecay, Tensor weights) {
    return RealScalar.ZERO;
  }

  @Override
  public Tensor gradient(boolean weightDecay, Tensor weights) {
    return weights.maps(Scalar::zero);
  }
}
