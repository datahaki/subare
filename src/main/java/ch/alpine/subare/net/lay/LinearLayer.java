// code by jph
package ch.alpine.subare.net.lay;

import java.util.random.RandomGenerator;

import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Slash;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Unprotect;
import ch.alpine.tensor.alg.ConstantArray;
import ch.alpine.tensor.ext.PackageTestAccess;
import ch.alpine.tensor.io.MathematicaFormat;
import ch.alpine.tensor.red.Total;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/LinearLayer.html">LinearLayer</a> */
public class LinearLayer extends MatrixLayer {
  /** Dimensions: d_out */
  @PackageTestAccess
  Tensor b;
  // ---
  /** Dimensions: d_out */
  private transient Tensor gb;

  public LinearLayer(int d_in, int d_out) {
    super(d_in, d_out);
    b = ConstantArray.of(DoubleScalar.INDETERMINATE, d_out);
  }

  @Override
  public Tensor forward(Tensor x) {
    return outputCache = super.forward(x).add(b);
  }

  @Override
  public Tensor forwardBatch(Tensor X) {
    return outputCache = Slash.of(super.forwardBatch(X), b::add);
  }

  /** Backward pass: computes gradients with respect to weights, bias, and i[...]
   * - grad_output shape: (batch_size, out_features) -> dL/dY */
  @Override
  public Tensor back(Tensor gradOutput) {
    gb = gradOutput;
    return super.back(gradOutput);
  }

  @Override
  public Tensor backBatch(Tensor gradOutput) {
    // Dimensions[gb] = d_out
    gb = Total.of(gradOutput);
    return super.backBatch(gradOutput);
  }

  @Override
  public void initialize(RandomGenerator randomGenerator) {
    super.initialize(randomGenerator);
    b = b.maps(Scalar::zero);
  }

  @Override // from ParametricLayer
  public Tensor parameters() {
    return Unprotect.byRef(super.parameters(), b);
  }

  @Override // from ParametricLayer
  public Tensor gradient(boolean decayInLoss) {
    return Unprotect.byRef(super.gradient(decayInLoss), gb);
  }

  @Override // from ParametricLayer
  public void increment(boolean decayInLoss, Tensor delta, Scalar eta) {
    super.increment(decayInLoss, delta.get(0), eta);
    b = b.add(delta.get(1).multiply(eta.negate()));
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("LinearLayer", W, b, netRegularization);
  }
}
