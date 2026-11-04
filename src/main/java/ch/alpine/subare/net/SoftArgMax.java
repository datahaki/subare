// code by jph
package ch.alpine.subare.net;

import java.io.Serializable;
import java.util.random.RandomGenerator;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.UnitVector;
import ch.alpine.tensor.ext.ArgMax;
import ch.alpine.tensor.nrm.SoftmaxVector;

public class SoftArgMax implements NetLayer, Serializable {
  Tensor interCache;

  @Override
  public void initialize(RandomGenerator randomGenerator) {
    interCache = null;
  }

  @Override
  public Scalar forward(Tensor x) {
    return RealScalar.of(ArgMax.of(interCache = SoftmaxVector.of(x)));
  }

  @Override
  public Tensor back(Tensor d) {
    return d;
  }

  @Override
  public Tensor error(Tensor y) {
    int k = Scalars.intValueExact((Scalar) y);
    return UnitVector.of(interCache.length(), k).subtract(interCache); // one-hot target
  }
}
