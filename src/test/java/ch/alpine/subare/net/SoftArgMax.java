// code by jph
package ch.alpine.subare.net;

import java.io.Serializable;
import java.util.random.RandomGenerator;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.VectorQ;
import ch.alpine.tensor.ext.ArgMax;
import ch.alpine.tensor.nrm.SoftmaxVector;

class SoftArgMax implements NetLayer, Serializable {
  private Tensor probs;

  @Override
  public void initialize(RandomGenerator randomGenerator) {
    probs = null;
  }

  @Override
  public Tensor forward(Tensor x) {
    if (VectorQ.of(x))
      return RealScalar.of(ArgMax.of(probs = SoftmaxVector.of(x)));
    // ---
    probs = Tensor.of(x.stream().map(SoftmaxVector::of));
    return Tensor.of(probs.stream().map(r -> RealScalar.of(ArgMax.of(r))));
  }

  @Override
  public NetError error(Tensor target) {
    return CrossEntropyLoss.of(probs, target);
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    return gradOutput;
  }
}
