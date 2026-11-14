// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;

import ch.alpine.subare.net.err.CrossEntropyLoss;
import ch.alpine.subare.net.err.NetLoss;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.VectorQ;
import ch.alpine.tensor.ext.ArgMax;

/** maps vector of probabilities to the ArgMax */
public class ArgMaxLayer implements NetLayer, Serializable {
  private Tensor inputCache;

  @Override
  public Tensor forward(Tensor x) {
    inputCache = x;
    return VectorQ.of(x) //
        ? Tensors.vector(ArgMax.of(x))
        : Tensor.of(x.stream().map(row -> Tensors.vector(ArgMax.of(row))));
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
}
