// code by jph
package ch.alpine.subare.net;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.api.ScalarUnaryOperator;
import ch.alpine.tensor.io.MathematicaFormat;
import ch.alpine.tensor.red.Entrywise;
import ch.alpine.tensor.sca.Ramp;
import ch.alpine.tensor.sca.UnitStep;
import ch.alpine.tensor.sca.exp.DLogisticSigmoid;
import ch.alpine.tensor.sca.exp.LogisticSigmoid;
import ch.alpine.tensor.sca.tri.Tanh;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/ElementwiseLayer.html">ElementwiseLayer</a> */
public abstract class ElementwiseLayer implements Layer {
  public static Layer logSig() {
    return new ElementwiseLayer(LogisticSigmoid.FUNCTION) {
      @Override
      public Tensor back(Tensor gradOutput) {
        return Entrywise.mul().apply(gradOutput, outputCache.maps(DLogisticSigmoid.NESTED));
      }

      @Override
      public String toString() {
        return MathematicaFormat.concise("LogisticSigmoidLayer");
      }
    };
  }

  /** ReLU:
   * gradInput[i] = (inputCache[i] > 0) ? gradOutput[i] : 0.0;
   * No shrinkage for positive inputs
   * Can “kill” neurons for negative inputs
   * 
   * @return */
  public static Layer relu() {
    return new ElementwiseLayer(Ramp.FUNCTION) {
      @Override
      public Tensor back(Tensor gradOutput) {
        return Entrywise.mul().apply(gradOutput, inputCache.maps(UnitStep.FUNCTION));
      }

      @Override
      public String toString() {
        return MathematicaFormat.concise("ReLuLayer");
      }
    };
  }

  /** Tanh:
   * gradInput[i] = gradOutput[i] * (1 - outputCache[i] * outputCache[i]);
   * Gradient shrinks near ±1
   * Can cause vanishing gradients
   * 
   * Tanh behaves best when:
   * 
   * input∼N(0,1)
   * 
   * @return */
  public static Layer tanh() {
    return new ElementwiseLayer(Tanh.FUNCTION) {
      private final ScalarUnaryOperator df = z -> z.one().subtract(z.multiply(z));

      @Override
      public Tensor back(Tensor gradOutput) {
        return Entrywise.mul().apply(gradOutput, outputCache.maps(df));
      }

      @Override
      public String toString() {
        return MathematicaFormat.concise("TanhLayer");
      }
    };
  }

  private final ScalarUnaryOperator f;
  /** output cache */
  Tensor inputCache;
  Tensor outputCache;

  public ElementwiseLayer(ScalarUnaryOperator f) {
    this.f = f;
  }

  @Override
  public Tensor forward(Tensor input) {
    return this.outputCache = ((inputCache = input).maps(f));
  }

  @Override
  public void update() {
  }

  @Override
  public Tensor error(Tensor y) {
    throw new IllegalStateException();
  }
}
