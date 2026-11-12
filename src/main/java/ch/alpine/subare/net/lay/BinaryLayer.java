// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;

import ch.alpine.subare.net.err.MSELoss;
import ch.alpine.subare.net.err.NetError;
import ch.alpine.tensor.Tensor;

/** LogisticSigmoid */
public class BinaryLayer implements NetLayer, Serializable {
  Tensor outputCache;

  @Override
  public Tensor forward(Tensor x) {
    return this.outputCache = x;
  }

  @Override
  public NetError error(Tensor target) {
    return MSELoss.of(outputCache, target);
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    return gradOutput;
  }
}
