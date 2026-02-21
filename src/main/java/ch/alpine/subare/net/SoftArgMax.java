// code by jph
package ch.alpine.subare.net;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.UnitVector;
import ch.alpine.tensor.ext.ArgMax;
import ch.alpine.tensor.nrm.SoftmaxVector;

public class SoftArgMax implements Layer {
  Tensor interCache;

  @Override
  public Scalar forward(Tensor x) {
    return RealScalar.of(ArgMax.of(interCache = SoftmaxVector.of(x)));
  }

  @Override
  public Tensor back(Tensor d) {
    return d;
  }

  @Override
  public void update() {
  }

  @Override
  public Tensor error(Tensor y) {
    int k = Scalars.intValueExact((Scalar) y);
    return UnitVector.of(interCache.length(), k).subtract(interCache); // one-hot target
  }

  @Override
  public Tensor parameters() {
    return Tensors.empty();
  }
}
