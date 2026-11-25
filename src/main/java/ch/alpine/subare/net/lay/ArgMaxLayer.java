// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;

import ch.alpine.subare.net.los.AbstractCrossEntropyLoss;
import ch.alpine.subare.net.los.LossFunction;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Slash;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.api.TensorUnaryOperator;
import ch.alpine.tensor.ext.ArgMax;

/** maps vector of probabilities to the ArgMax */
public class ArgMaxLayer implements NetLayer, Serializable {
  private static final TensorUnaryOperator FORWARD = x -> RealScalar.of(ArgMax.of(x));
  private transient Tensor probabilities;

  /** {@inheritDoc}
   * 
   * @param x required to be a vector
   * @return vector of length 1 with entry as non-negative integer */
  @Override
  public Tensor forward(Tensor x) {
    return FORWARD.apply(probabilities = x);
  }

  @Override
  public Tensor forwardBatch(Tensor X) {
    return Slash.of(probabilities = X, FORWARD);
  }

  public LossFunction crossEntropyLoss() {
    return new AbstractCrossEntropyLoss() {
      @Override
      protected Tensor probabilities() {
        return probabilities;
      }
    };
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
