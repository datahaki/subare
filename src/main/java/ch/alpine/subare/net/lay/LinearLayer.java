// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;
import java.util.List;
import java.util.random.RandomGenerator;
import java.util.stream.Stream;

import ch.alpine.subare.net.err.MSELoss;
import ch.alpine.subare.net.err.NetError;
import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.Rational;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.alg.ConstantArray;
import ch.alpine.tensor.alg.Dimensions;
import ch.alpine.tensor.alg.Flatten;
import ch.alpine.tensor.alg.VectorQ;
import ch.alpine.tensor.io.MathematicaFormat;
import ch.alpine.tensor.lie.TensorProduct;
import ch.alpine.tensor.mat.MatrixDotTranspose;
import ch.alpine.tensor.mat.TransposeMatrixDot;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;
import ch.alpine.tensor.red.Total;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/LinearLayer.html">LinearLayer</a> */
public class LinearLayer implements NetLayer, Serializable {
  /** Dimensions: d_in x d_out */
  public Tensor W;
  /** Dimensions: d_out */
  public Tensor b;
  /** Dimensions: d_in, or batch_size x d_in */
  private Tensor inputCache;
  /** Dimensions: d_out, or batch_size x d_out */
  private Tensor outputCache;
  // For each parameter, the optimizer maintains its own velocity.
  /** Dimensions: d_in x d_out */
  public Tensor gW;
  /** Dimensions: d_out */
  public Tensor gb;

  public LinearLayer(int d_in, int d_out) {
    W = ConstantArray.of(DoubleScalar.INDETERMINATE, d_in, d_out);
    b = ConstantArray.of(DoubleScalar.INDETERMINATE, d_out);
  }

  public void initialize(RandomGenerator randomGenerator) {
    List<Integer> dims = Dimensions.of(W);
    int d_in = dims.get(0);
    int d_out = dims.get(1);
    Distribution distribution = NormalDistribution.of(RealScalar.ZERO, Rational.of(2, d_in));
    W = RandomVariate.of(distribution, randomGenerator, dims);
    b = Array.zeros(d_out);
  }

  @Override
  public Tensor forward(Tensor x) {
    inputCache = x;
    return outputCache = VectorQ.of(x) //
        ? x.dot(W).add(b)
        : Tensor.of(x.dot(W).stream().map(b::add)); // Dimensions: batch_size x d_out
  }

  /** Backward pass: computes gradients with respect to weights, bias, and i[...]
   * - grad_output shape: (batch_size, out_features) -> dL/dY */
  @Override
  public Tensor back(Tensor gradOutput) {
    // Dimensions[gradOutput] = batch_size x d_out
    if (VectorQ.of(gradOutput)) {
      gW = TensorProduct.of(inputCache, gradOutput);
      gb = gradOutput;
      return W.dot(gradOutput); // vector of length d_in
    }
    /** requires gradOutput to be prescaled with 1/batch_size
     * 
     * Note: If your loss function divides by batch_size (mean reduction),
     * grad_output already contains the 1/B factor, so this matrix product
     * automatically yields the mean gradient for the batch. */
    // Dimensions[gW] = d_in x d_out
    gW = TransposeMatrixDot.of(inputCache, gradOutput);
    // Dimensions[gb] = d_out
    gb = Total.of(gradOutput);
    // Dimensions[gradInput] = batch_size x d_in
    return MatrixDotTranspose.of(gradOutput, W);
    // Sign.requirePositiveOrZero(l2);
    // if (Scalars.nonZero(l2))
    // gW = gW.subtract(W.multiply(l2));
  }

  @Override
  public NetError error(Tensor target) {
    return MSELoss.of(outputCache, target); // vector of length d_out
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("LinearLayer", W, b);
  }
}
