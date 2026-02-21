// code by jph
package ch.alpine.subare.net;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.red.Entrywise;
import ch.alpine.tensor.sca.exp.DLogisticSigmoid;

public class BinaryLayer implements Layer {
  Tensor x;

  @Override
  public Tensor forward(Tensor x) {
    return this.x = x;
  }

  @Override
  public Tensor back(Tensor d) {
    return Entrywise.mul().apply(d, x.maps(DLogisticSigmoid.NESTED));
  }

  @Override
  public void update() {
  }

  @Override
  public Tensor error(Tensor y) {
    // y should be either 0 or 1
    return y.subtract(x);
  }

  @Override
  public Tensor parameters() {
    return Tensors.empty();
  }
}
