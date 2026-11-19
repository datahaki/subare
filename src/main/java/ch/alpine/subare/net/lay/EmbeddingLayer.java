// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;
import java.util.List;
import java.util.random.RandomGenerator;

import ch.alpine.subare.net.api.NetRegularization;
import ch.alpine.subare.net.api.ParametricLayer;
import ch.alpine.subare.net.reg.L2Regularization;
import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.Rational;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.ConstantArray;
import ch.alpine.tensor.alg.Dimensions;
import ch.alpine.tensor.alg.VectorQ;
import ch.alpine.tensor.ext.Int;
import ch.alpine.tensor.ext.Integers;
import ch.alpine.tensor.ext.PackageTestAccess;
import ch.alpine.tensor.io.MathematicaFormat;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;

/** Embeds integers between 0 and v-1 into a continuous vector space of dimension d.
 * 
 * Backpropagation stops at the embedding layer!
 * 
 * inspired by
 * <a href="https://reference.wolfram.com/language/ref/EmbeddingLayer.html">EmbeddingLayer</a> */
public class EmbeddingLayer implements ParametricLayer, Serializable {
  @PackageTestAccess
  Tensor W;
  // ---
  public NetRegularization netRegularization = new L2Regularization(RealScalar.of(0.001));
  /** Dimensions: d_in, or batch_size x d_in */
  private transient Tensor inputCache;
  /** Dimensions: d_in x d_out */
  private transient Tensor gW;

  /** @param v vocabulary size (total number of unique tokens)
   * @param d embedding dimension */
  public EmbeddingLayer(int v, int d) {
    Integers.requirePositive(v);
    Integers.requirePositive(d);
    W = ConstantArray.of(DoubleScalar.INDETERMINATE, v, d);
  }

  @Override
  public Tensor forward(Tensor x) {
    inputCache = VectorQ.require(x);
    // returns dimensions: [x.length() * d]
    return Tensor.of(x.stream() //
        .map(Scalar.class::cast) //
        .mapToInt(Scalars::intValueExact) //
        .mapToObj(W::get));
  }

  @Override
  public Tensor forwardBatch(Tensor X) {
    Tensor Y = Tensor.of(X.stream().map(this::forward));
    inputCache = X; // override
    return Y;
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    gW = W.maps(Scalar::zero);
    contrib(gW, gradOutput, inputCache);
    return null;
  }

  private static void contrib(Tensor gW, Tensor gradOutput, Tensor inputCache) {
    Int myI = new Int();
    gradOutput.stream() //
        .forEach(row -> gW.set(row::add, inputCache.Get(myI.getAndIncrement()).number().intValue()));
  }

  @Override
  public Tensor backBatch(Tensor gradOutput) {
    gW = W.maps(Scalar::zero);
    int n = gradOutput.length();
    Integers.requireEquals(n, inputCache.length());
    for (int i = 0; i < n; ++i)
      contrib(gW, gradOutput.get(i), inputCache.get(i));
    return null;
  }

  @Override
  public void initialize(RandomGenerator randomGenerator) {
    List<Integer> dims = Dimensions.of(W);
    int d = dims.get(1);
    Distribution distribution = NormalDistribution.of(RealScalar.ZERO, Rational.of(1, d));
    W = RandomVariate.of(distribution, randomGenerator, dims);
  }

  @Override
  public Tensor parameters() {
    return W;
  }

  @Override
  public Tensor gradient(boolean decayInLoss) {
    return gW.add(netRegularization.gradient(decayInLoss, W));
  }

  @Override
  public void increment(boolean decayInLoss, Tensor delta, Scalar eta) {
    Scalar eta_negate = eta.negate();
    Tensor dW = delta.add(netRegularization.sneakDecay(decayInLoss, W));
    W = W.add(dW.multiply(eta_negate));
  }

  @Override
  public Scalar reg_loss(boolean weightDecay) {
    return netRegularization.penalty(weightDecay, W);
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("EmbeddingLayer", W, netRegularization);
  }
}
