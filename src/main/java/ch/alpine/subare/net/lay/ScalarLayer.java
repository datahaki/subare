// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;

import ch.alpine.subare.net.LossFunction;
import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.NetLoss;
import ch.alpine.subare.net.err.SquaredErrorLoss;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Slash;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.VectorQ;

/** typically last layer in network following
 * ElementwiseLayers.SIGMOID.get()
 * used for binary decision tasks, i.e. true or false
 * 
 * the output for {@link #forward(Tensor)} is of
 * type {@link Scalar} */
public class ScalarLayer implements NetLayer, LossFunction, Serializable {
  transient Tensor inputCache;

  @Override
  public Tensor forward(Tensor x) {
    VectorQ.requireLength(x, 1);
    inputCache = x;
    return x.Get(0);
  }

  @Override
  public Tensor forwardBatch(Tensor X) {
    inputCache = X;
    return Slash.of(X, v -> v.Get(0));
  }

  @Override
  public NetLoss compute(Tensor prediction, Tensor target) {
    Scalar scalar = (Scalar) target;
    return SquaredErrorLoss.INSTANCE.compute(inputCache, Tensors.of(scalar));
  }

  @Override
  public NetLoss computeBatch(Tensor prediction, Tensor target) {
    return SquaredErrorLoss.INSTANCE.computeBatch(inputCache, Slash.of(target, Tensors::of));
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
    return "ScalarLayer";
  }
}
