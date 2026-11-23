// code by jph
package ch.alpine.subare.net.lay;

import ch.alpine.subare.net.NetLayer;
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
public enum ScalarLayer implements NetLayer {
  VectorToScalar {
    @Override
    public Tensor forward(Tensor x) {
      VectorQ.requireLength(x, 1);
      return x.Get(0);
    }

    @Override
    public Tensor back(Tensor gradOutput) {
      return Tensors.of(gradOutput);
    }
  },
  ScalarToVector {
    @Override
    public Tensor forward(Tensor x) {
      return Tensors.of(x);
    }

    @Override
    public Tensor back(Tensor gradOutput) {
      return gradOutput.Get(0);
    }
  };

  @Override
  public final Tensor forwardBatch(Tensor X) {
    return Slash.of(X, this::forward);
  }

  @Override
  public final Tensor backBatch(Tensor gradOutput) {
    return Slash.of(gradOutput, this::back);
  }
}
