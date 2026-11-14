// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;

import ch.alpine.subare.net.err.NetLoss;
import ch.alpine.subare.net.err.SquaredErrorLoss;
import ch.alpine.tensor.Tensor;

/** LogisticSigmoid */
public class BinaryLayer implements NetLayer, Serializable {
  Tensor outputCache;

  @Override
  public Tensor forward(Tensor x) {
    return this.outputCache = x;
  }

  @Override
  public NetLoss error(Tensor target) {
    return SquaredErrorLoss.of(outputCache, target);
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    return gradOutput;
  }
}
