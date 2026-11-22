// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;

import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.NetLoss;
import ch.alpine.subare.net.NetOutput;
import ch.alpine.subare.net.err.CrossEntropyLoss;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Slash;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.api.TensorUnaryOperator;
import ch.alpine.tensor.ext.ArgMax;

/** maps vector of probabilities to the ArgMax */
public class ArgMaxLayer implements NetLayer, NetOutput, Serializable {
  private static final TensorUnaryOperator FORWARD = x -> RealScalar.of(ArgMax.of(x));
  private transient Tensor inputCache;

  /** {@inheritDoc}
   * 
   * @param x required to be a vector
   * @return vector of length 1 with entry as non-negative integer */
  @Override
  public Tensor forward(Tensor x) {
    return FORWARD.apply(inputCache = x);
  }

  @Override
  public Tensor forwardBatch(Tensor X) {
    return Slash.of(inputCache = X, FORWARD);
  }

  /** @param target */
  @Override
  public NetLoss error(Tensor target) {
    return CrossEntropyLoss.of(inputCache, target);
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
    return "ArgMaxLayer";
  }
}
