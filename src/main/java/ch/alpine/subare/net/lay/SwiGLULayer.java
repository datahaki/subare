// code by gemini
package ch.alpine.subare.net.lay;

import java.io.Serializable;
import java.util.LinkedList;
import java.util.List;
import java.util.random.RandomGenerator;

import ch.alpine.subare.net.TrainableNetLayer;
import ch.alpine.subare.net.reg.L2Regularization;
import ch.alpine.subare.net.reg.NetRegularization;
import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.Rational;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.Unprotect;
import ch.alpine.tensor.alg.ConstantArray;
import ch.alpine.tensor.alg.Dimensions;
import ch.alpine.tensor.alg.VectorQ;
import ch.alpine.tensor.ext.Integers;
import ch.alpine.tensor.ext.PackageTestAccess;
import ch.alpine.tensor.io.MathematicaFormat;
import ch.alpine.tensor.lie.TensorProduct;
import ch.alpine.tensor.mat.MatrixDotTranspose;
import ch.alpine.tensor.mat.TransposeMatrixDot;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;
import ch.alpine.tensor.red.Entrywise;
import ch.alpine.tensor.sca.exp.LogisticSigmoid;
import ch.alpine.tensor.sca.pow.Sqrt;

/** Swish-Gated Linear Unit layer without bias:
 * Y = (Swish(X * W_gate) * (X * W_val)) * W_out */
public class SwiGLULayer implements TrainableNetLayer, Serializable {
  /** Dimensions: d_in x d_hidden */
  @PackageTestAccess
  Tensor W_gate;
  /** Dimensions: d_in x d_hidden */
  @PackageTestAccess
  Tensor W_val;
  /** Dimensions: d_hidden x d_out */
  @PackageTestAccess
  Tensor W_out;
  // ---
  public NetRegularization netRegularization = new L2Regularization(RealScalar.of(0.001));
  // ---
  private transient Tensor inputCache;
  private transient Tensor gateCache;
  private transient Tensor sigGateCache;
  private transient Tensor swishGateCache;
  private transient Tensor valCache;
  private transient Tensor hiddenCache;
  // ---
  private transient Tensor gW_gate;
  private transient Tensor gW_val;
  private transient Tensor gW_out;

  public SwiGLULayer(int d_in, int d_hidden, int d_out) {
    Integers.requirePositive(d_in);
    Integers.requirePositive(d_hidden);
    Integers.requirePositive(d_out);
    W_gate = ConstantArray.of(DoubleScalar.INDETERMINATE, d_in, d_hidden);
    W_val = ConstantArray.of(DoubleScalar.INDETERMINATE, d_in, d_hidden);
    W_out = ConstantArray.of(DoubleScalar.INDETERMINATE, d_hidden, d_out);
  }

  @Override
  public Tensor forward(Tensor x) {
    return forwardBatch(VectorQ.require(x));
  }

  @Override
  public Tensor forwardBatch(Tensor X) {
    inputCache = X;
    gateCache = X.dot(W_gate);
    valCache = X.dot(W_val);
    sigGateCache = gateCache.maps(LogisticSigmoid.FUNCTION);
    swishGateCache = Entrywise.mul().apply(gateCache, sigGateCache);
    hiddenCache = Entrywise.mul().apply(swishGateCache, valCache);
    return hiddenCache.dot(W_out);
  }

  private Tensor swishBackward(Tensor gradSwish) {
    // d/dg [g * sigmoid(g)] = sigmoid(g) * (1 + g * (1 - sigmoid(g)))
    Tensor oneMinusSig = sigGateCache.maps(s -> RealScalar.ONE.subtract(s));
    Tensor inner = Entrywise.mul().apply(gateCache, oneMinusSig).maps(RealScalar.ONE::add);
    Tensor swishDeriv = Entrywise.mul().apply(sigGateCache, inner);
    return Entrywise.mul().apply(gradSwish, swishDeriv);
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    VectorQ.require(gradOutput);
    gW_out = TensorProduct.of(hiddenCache, gradOutput);
    Tensor gradHidden = W_out.dot(gradOutput);
    Tensor gradVal = Entrywise.mul().apply(gradHidden, swishGateCache);
    Tensor gradSwish = Entrywise.mul().apply(gradHidden, valCache);
    Tensor gradGate = swishBackward(gradSwish);
    gW_gate = TensorProduct.of(inputCache, gradGate);
    gW_val = TensorProduct.of(inputCache, gradVal);
    return W_gate.dot(gradGate).add(W_val.dot(gradVal));
  }

  @Override
  public Tensor backBatch(Tensor gradOutput) {
    gW_out = TransposeMatrixDot.of(hiddenCache, gradOutput);
    Tensor gradHidden = MatrixDotTranspose.of(gradOutput, W_out);
    Tensor gradVal = Entrywise.mul().apply(gradHidden, swishGateCache);
    Tensor gradSwish = Entrywise.mul().apply(gradHidden, valCache);
    Tensor gradGate = swishBackward(gradSwish);
    gW_gate = TransposeMatrixDot.of(inputCache, gradGate);
    gW_val = TransposeMatrixDot.of(inputCache, gradVal);
    return MatrixDotTranspose.of(gradGate, W_gate).add(MatrixDotTranspose.of(gradVal, W_val));
  }

  private static Tensor sampleNormal(RandomGenerator randomGenerator, List<Integer> dims) {
    int fanIn = dims.get(0);
    Scalar sigma = Sqrt.FUNCTION.apply(Rational.of(2, fanIn));
    Distribution distribution = NormalDistribution.of(RealScalar.ZERO, sigma);
    return RandomVariate.of(distribution, randomGenerator, dims);
  }

  @Override
  public void initialize(RandomGenerator randomGenerator) {
    W_gate = sampleNormal(randomGenerator, Dimensions.of(W_gate));
    W_val = sampleNormal(randomGenerator, Dimensions.of(W_val));
    W_out = sampleNormal(randomGenerator, Dimensions.of(W_out));
  }

  @Override // from ParametricLayer
  public Tensor parameters() {
    return Unprotect.byRef(W_gate, W_val, W_out);
  }

  @Override // from ParametricLayer
  public Tensor gradient(boolean decayInLoss) {
    return Tensors.of( //
        gW_gate.add(netRegularization.gradient(decayInLoss, W_gate)), //
        gW_val.add(netRegularization.gradient(decayInLoss, W_val)), //
        gW_out.add(netRegularization.gradient(decayInLoss, W_out)));
  }

  private Tensor applyWeightStep(Tensor weight, Tensor deltaSlice, Scalar etaNegate, boolean decayInLoss) {
    Tensor dw = deltaSlice.add(netRegularization.sneakDecay(decayInLoss, weight));
    return weight.add(dw.multiply(etaNegate));
  }

  @Override // from ParametricLayer
  public void step(boolean decayInLoss, Tensor delta, Scalar eta) {
    Scalar etaNegate = eta.negate();
    W_gate = applyWeightStep(W_gate, delta.get(0), etaNegate, decayInLoss);
    W_val = applyWeightStep(W_val, delta.get(1), etaNegate, decayInLoss);
    W_out = applyWeightStep(W_out, delta.get(2), etaNegate, decayInLoss);
  }

  @Override
  public final Scalar reg_loss(boolean weightDecay) {
    return netRegularization.penalty(weightDecay, W_gate) //
        .add(netRegularization.penalty(weightDecay, W_val)) //
        .add(netRegularization.penalty(weightDecay, W_out));
  }

  @Override
  public String toString() {
    List<Integer> dims = new LinkedList<>();
    dims.add(W_gate.length());
    dims.addAll(Dimensions.of(W_out));
    return MathematicaFormat.concise( //
        "SwiGLULayer", //
        dims, //
        netRegularization);
  }
}
