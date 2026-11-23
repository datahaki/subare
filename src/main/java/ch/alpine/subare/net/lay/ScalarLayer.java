// code by jph
package ch.alpine.subare.net.lay;

import java.util.List;

import ch.alpine.subare.net.NetLayer;
import ch.alpine.tensor.Slash;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Unprotect;
import ch.alpine.tensor.alg.VectorQ;

public enum ScalarLayer implements NetLayer {
  VectorToScalar {
    @Override
    public Tensor forward(Tensor x) {
      VectorQ.requireLength(x, 1);
      return x.Get(0);
    }

    @Override
    public Tensor back(Tensor gradOutput) {
      return Unprotect.using(List.of(gradOutput));
    }
  },
  ScalarToVector {
    @Override
    public Tensor forward(Tensor x) {
      return Unprotect.using(List.of(x));
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
