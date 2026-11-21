// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;

import ch.alpine.subare.net.api.NetLayer;
import ch.alpine.subare.net.api.NetLoss;
import ch.alpine.subare.net.err.CrossEntropyLoss;
import ch.alpine.tensor.Slash;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.VectorQ;
import ch.alpine.tensor.ext.ArgMax;

/** maps vector of probabilities to the ArgMax */
public class ArgMaxLayer implements NetLayer, Serializable {
  private transient Tensor inputCache;

  @Override
  public Tensor forward(Tensor x) {
    inputCache = VectorQ.require(x); // TODO VectorQ check is obsolete
    return Tensors.vector(ArgMax.of(x));
  }

  @Override
  public Tensor forwardBatch(Tensor X) {
    inputCache = X;
    return Slash.of(X, x -> Tensors.vector(ArgMax.of(x))); // ideally call this::forward
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
}
