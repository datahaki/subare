// code by jph
package ch.alpine.subare.net;

import java.io.Serializable;
import java.util.random.RandomGenerator;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
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
  public NetError error(Tensor target) {
    return CrossEntropyLoss.of(interCache, target);
  }
}
