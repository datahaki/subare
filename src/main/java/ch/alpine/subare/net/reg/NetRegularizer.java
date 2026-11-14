// code by jph
package ch.alpine.subare.net.reg;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

public interface NetRegularizer {
  final NetRegularizer NONE = new NetRegularizer() {
    @Override
    public Scalar penalty(Tensor weights) {
      return RealScalar.ZERO;
    }

    @Override
    public Tensor gradient(Tensor weights) {
      return weights.maps(Scalar::zero);
    }
  };

  Scalar penalty(Tensor weights);

  Tensor gradient(Tensor weights);
}
