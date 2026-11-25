// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;

import ch.alpine.subare.net.LossFunction;
import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.err.AbstractBinaryCrossEntropy;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.sca.UnitStep;

public class UnitStepLayer implements NetLayer, Serializable {
  private transient Tensor logits;

  /** @param x required to be a scalar
   * @return integer scalar 0 or 1 */
  @Override
  public Tensor forward(Tensor x) {
    return (logits = x).maps(UnitStep.FUNCTION);
  }

  @Override
  public Tensor forwardBatch(Tensor X) {
    return (logits = X).maps(UnitStep.FUNCTION);
  }

  public LossFunction lossFunction() {
    return new AbstractBinaryCrossEntropy() {
      @Override
      protected Tensor logits() {
        return logits;
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
    return "UnitStepLayer";
  }
}
