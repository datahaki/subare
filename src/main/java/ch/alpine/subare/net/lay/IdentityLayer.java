// code by jph
package ch.alpine.subare.net.lay;

import ch.alpine.subare.net.api.NetLayer;
import ch.alpine.tensor.Tensor;

public enum IdentityLayer implements NetLayer {
  INSTANCE;

  @Override
  public Tensor forward(Tensor x) {
    return x;
  }

  @Override
  public Tensor forwardBatch(Tensor X) {
    return X;
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    return gradOutput;
  }

  @Override
  public Tensor backBatch(Tensor gradOutput) {
    return gradOutput;
  }

  @Override
  public String toString() {
    return "IdentityLayer";
  }
}
