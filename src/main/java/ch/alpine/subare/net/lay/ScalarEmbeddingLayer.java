// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;
import java.util.List;
import java.util.random.RandomGenerator;

import ch.alpine.subare.net.api.NetLayer;
import ch.alpine.subare.net.api.NetRegularization;
import ch.alpine.subare.net.reg.L2Regularization;
import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.Rational;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Slash;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.ConstantArray;
import ch.alpine.tensor.alg.Dimensions;
import ch.alpine.tensor.ext.Int;
import ch.alpine.tensor.ext.Integers;
import ch.alpine.tensor.ext.PackageTestAccess;
import ch.alpine.tensor.io.MathematicaFormat;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;

/** Embeds integers between 0 and (v-1) into a continuous vector space of dimension d.
 * 
 * Backpropagation stops at the embedding layer!
 * 
 * inspired by
 * <a href="https://reference.wolfram.com/language/ref/EmbeddingLayer.html">EmbeddingLayer</a> */
// TODO misnomer
public class ScalarEmbeddingLayer implements NetLayer, Serializable {
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
  public ScalarEmbeddingLayer(int v, int d) {
    Integers.requirePositive(v);
    Integers.requirePositive(d);
    W = ConstantArray.of(DoubleScalar.INDETERMINATE, v, d);
  }

  @Override
  public Tensor forward(Tensor x) {
    Integers.requireEquals(x.length(), 1);
    inputCache = x;
    // returns dimensions: [x.length() * d]
    int index = Scalars.intValueExact(x.Get(0));
    return W.get(index);
  }

  @Override
  public Tensor forwardBatch(Tensor X) {
    Tensor Y = Slash.of(X, this::forward);
    inputCache = X; // override
    return Y;
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    int index = Scalars.intValueExact(inputCache.Get(0));
    gW = W.maps(Scalar::zero);
    gW.set(gradOutput::add, index); // TODO simplify later
    return null;
  }

  @Override
  public Tensor backBatch(Tensor gradOutput) {
    gW = W.maps(Scalar::zero);
    int n = gradOutput.length();
    Integers.requireEquals(n, inputCache.length());
    Int myI = new Int();
    gradOutput.stream() //
        .forEach(row -> gW.set(row::add, inputCache.Get(myI.getAndIncrement(), 0).number().intValue()));
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
