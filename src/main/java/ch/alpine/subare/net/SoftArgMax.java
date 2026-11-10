// code by jph
package ch.alpine.subare.net;

import java.io.Serializable;
import java.util.random.RandomGenerator;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Unprotect;
import ch.alpine.tensor.alg.UnitVector;
import ch.alpine.tensor.alg.VectorQ;
import ch.alpine.tensor.ext.ArgMax;
import ch.alpine.tensor.nrm.SoftmaxVector;

// TODO breakup into two classes
public class SoftArgMax implements NetLayer, Serializable {
  Tensor interCache;

  @Override
  public void initialize(RandomGenerator randomGenerator) {
    interCache = null;
  }

  @Override
  public Tensor forward(Tensor x) {
    if (VectorQ.of(x))
      return RealScalar.of(ArgMax.of(interCache = SoftmaxVector.of(x)));
    // ---
    interCache = Tensor.of(x.stream().map(SoftmaxVector::of));
    return Tensor.of(interCache.stream().map(r -> RealScalar.of(ArgMax.of(r))));
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    return gradOutput;
  }

  @Override
  public Tensor error(Tensor y) {
    Tensor target;
    if (y instanceof Scalar) {
      int k = Scalars.intValueExact((Scalar) y);
      // TODO not efficient to build unit vector explicitly
      target = UnitVector.of(interCache.length(), k); // one-hot target
    } else {
      int cc = Unprotect.dimension1Hint(interCache);
      target = Tensor.of(y.stream() //
          .map(Scalar.class::cast) //
          .mapToInt(Scalars::intValueExact) //
          .mapToObj(k -> UnitVector.of(cc, k)));
    }
    return target.subtract(interCache);
  }
}
