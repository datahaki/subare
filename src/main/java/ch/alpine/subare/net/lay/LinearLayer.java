// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;
import java.util.List;
import java.util.random.RandomGenerator;

import ch.alpine.subare.net.err.NetLoss;
import ch.alpine.subare.net.err.SquaredErrorLoss;
import ch.alpine.subare.net.reg.L2Regularizer;
import ch.alpine.subare.net.reg.NetRegularizer;
import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.Rational;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Unprotect;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.alg.ConstantArray;
import ch.alpine.tensor.alg.Dimensions;
import ch.alpine.tensor.alg.VectorQ;
import ch.alpine.tensor.ext.PackageTestAccess;
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
public class LinearLayer implements ParametricLayer, Serializable {
  /** Dimensions: d_in x d_out */
  @PackageTestAccess
  Tensor W;
  /** Dimensions: d_out */
  @PackageTestAccess
  Tensor b;
  // ---
  public NetRegularizer netRegularizer = new L2Regularizer(RealScalar.of(0.001));
  /** Dimensions: d_in, or batch_size x d_in */
  private transient Tensor inputCache;
  /** Dimensions: d_out, or batch_size x d_out */
  private transient Tensor outputCache;
  // For each parameter, the optimizer maintains its own velocity.
  /** Dimensions: d_in x d_out */
  private transient Tensor gW;
  /** Dimensions: d_out */
  private transient Tensor gb;

  public LinearLayer(int d_in, int d_out) {
    W = ConstantArray.of(DoubleScalar.INDETERMINATE, d_in, d_out);
    b = ConstantArray.of(DoubleScalar.INDETERMINATE, d_out);
  }

  @Override // from ParametricLayer
  public synchronized Tensor parameters() {
    return Unprotect.byRef(W, b);
  }

  @Override // from ParametricLayer
  public Tensor gradient() {
    return Unprotect.byRef( //
        gW.add(netRegularizer.gradient(W)), //
        gb);
  }

  @Override // from ParametricLayer
  public synchronized void increment(Tensor delta) {
    W = W.add(delta.get(0));
    b = b.add(delta.get(1));
  }

  @Override
  public Tensor gradientWithoutDecay() {
    return Unprotect.byRef(gW, gb);
  }

  @Override
  public synchronized void incrementWithoutDecay(Tensor delta, Scalar eta) {
    Tensor dW = netRegularizer.gradient(W, eta.negate());
    W = W.add(delta.get(0)).add(dW);
    b = b.add(delta.get(1));
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
  public synchronized Tensor forward(Tensor x) {
    inputCache = VectorQ.require(x);
    return outputCache = x.dot(W).add(b);
  }

  @Override
  public synchronized Tensor forwardBatch(Tensor X) {
    inputCache = X;
    // Dimensions: batch_size x d_out
    return outputCache = Tensor.of(X.dot(W).stream().map(b::add));
  }

  /** Backward pass: computes gradients with respect to weights, bias, and i[...]
   * - grad_output shape: (batch_size, out_features) -> dL/dY */
  @Override
  public Tensor back(Tensor gradOutput) {
    VectorQ.require(gradOutput);
    gW = TensorProduct.of(inputCache, gradOutput);
    gb = gradOutput;
    return W.dot(gradOutput); // vector of length d_in
  }

  @Override
  public Tensor backBatch(Tensor gradOutput) {
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
  }

  public Scalar reg_loss() {
    return netRegularizer.penalty(W);
  }

  @Override
  public NetLoss error(Tensor target) {
    return SquaredErrorLoss.of(outputCache, target); // vector of length d_out
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("LinearLayer", W, b, netRegularizer);
  }
}
