// code by jph
package ch.alpine.subare.net;

import java.io.Serializable;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.red.Entrywise;
import ch.alpine.tensor.sca.exp.DLogisticSigmoid;

public class BinaryLayer implements NetLayer, Serializable {
  Tensor inputCache;

  @Override
  public Tensor forward(Tensor x) {
    return this.inputCache = x;
  }

  @Override
  public Tensor back(Tensor d) {
    return Entrywise.mul().apply(d, inputCache.maps(DLogisticSigmoid.NESTED));
  }

  @Override
  public Tensor error(Tensor y) {
    // y should be either 0 or 1
    return y.subtract(inputCache);
  }
}
