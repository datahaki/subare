// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;

import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.NetLoss;
import ch.alpine.subare.net.NetOutput;
import ch.alpine.subare.net.err.SquaredErrorLoss;
import ch.alpine.tensor.Tensor;

/** LogisticSigmoid */
public class BinaryLayer implements NetLayer, NetOutput, Serializable {
  transient Tensor outputCache;

  @Override
  public Tensor forward(Tensor x) {
    return outputCache = x;
  }

  @Override
  public Tensor forwardBatch(Tensor X) {
    return outputCache = X;
  }

  @Override
  public NetLoss error(Tensor target) {
    return SquaredErrorLoss.of(outputCache, target);
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
    return "BinaryLayer";
  }
}
