// code by gemini
package ch.alpine.subare.net.lay;

import java.io.Serializable;
import java.util.random.RandomGenerator;

import ch.alpine.subare.net.reg.NetRegularization;
import ch.alpine.subare.net.reg.NoRegularization;
import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.Parallelize;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Slash;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.ConstantArray;
import ch.alpine.tensor.alg.Dimensions;
import ch.alpine.tensor.alg.Flatten;
import ch.alpine.tensor.alg.VectorQ;
import ch.alpine.tensor.ext.Integers;
import ch.alpine.tensor.ext.PackageTestAccess;
import ch.alpine.tensor.io.MathematicaFormat;
import ch.alpine.tensor.lie.TensorProduct;
import ch.alpine.tensor.lie.rot.AngleVector;
import ch.alpine.tensor.mat.MatrixDotTranspose;
import ch.alpine.tensor.mat.TransposeMatrixDot;
import ch.alpine.tensor.num.Pi;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;
import ch.alpine.tensor.sca.Sign;
import ch.alpine.tensor.sca.tri.Cos;
import ch.alpine.tensor.sca.tri.Sin;

public class FourierLayer implements TrainableNetLayer, Serializable {
  private static final Scalar ETA_SCALE = RealScalar.of(0.07);
  /** Dimensions: d_in x numFreq */
  @PackageTestAccess
  Tensor B;
  // ---
  public NetRegularization netRegularization = NoRegularization.INSTANCE;
  private final Scalar sigma;
  private transient Tensor inputCache;
  private transient Tensor phaseCache;
  private transient Tensor gB;

  public FourierLayer(int d_in, int numFreq, Scalar sigma) {
    Integers.requirePositive(d_in);
    Integers.requirePositive(numFreq);
    this.sigma = Sign.requirePositive(sigma);
    B = ConstantArray.of(DoubleScalar.INDETERMINATE, d_in, numFreq);
  }

  @Override
  public void initialize(RandomGenerator randomGenerator) {
    B = RandomVariate.of( //
        NormalDistribution.of(sigma.zero(), sigma), //
        randomGenerator, //
        Dimensions.of(B)).multiply(Pi.TWO);
  }

  @Override
  public Tensor forward(Tensor x) {
    inputCache = VectorQ.require(x);
    phaseCache = x.dot(B);
    return Flatten.of(phaseCache.maps(AngleVector::of));
  }

  @Override
  public Tensor forwardBatch(Tensor X) {
    inputCache = X;
    phaseCache = Parallelize.dot(X, B);
    return Slash.of(phaseCache, row -> Flatten.of(row.maps(AngleVector::of)));
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    VectorQ.require(gradOutput);
    Tensor delta = phaseDelta(phaseCache, gradOutput);
    gB = TensorProduct.of(inputCache, delta);
    return B.dot(delta);
  }

  @Override
  public Tensor backBatch(Tensor gradOutput) {
    Tensor deltas = Slash.of(FourierLayer::phaseDelta, phaseCache, gradOutput);
    gB = TransposeMatrixDot.of(inputCache, deltas);
    return MatrixDotTranspose.of(deltas, B);
  }

  private static Tensor phaseDelta(Tensor phase, Tensor gradOut) {
    int numFreq = phase.length();
    return Tensors.vector(k -> {
      Scalar p = phase.Get(k);
      Scalar gCos = gradOut.Get(2 * k);
      Scalar gSin = gradOut.Get(2 * k + 1);
      // dot product of D.AngleVector (phase shift *90) and 2-excerpt from gradOutput
      return Sin.FUNCTION.apply(p).negate().multiply(gCos).add(Cos.FUNCTION.apply(p).multiply(gSin));
    }, numFreq);
  }

  @Override
  public Tensor parameters() {
    return B;
  }

  @Override
  public Tensor gradient(boolean decayInLoss) {
    return gB.add(netRegularization.gradient(decayInLoss, B));
  }

  @Override
  public void step(boolean decayInLoss, Tensor delta, Scalar eta) {
    Scalar eta_negate = eta.multiply(ETA_SCALE).negate();
    Tensor dB = delta.add(netRegularization.sneakDecay(decayInLoss, B));
    B = B.add(dB.multiply(eta_negate));
  }

  @Override
  public final Scalar reg_loss(boolean weightDecay) {
    return netRegularization.penalty(weightDecay, B);
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("FourierLayer", Dimensions.of(B), netRegularization, ETA_SCALE);
  }
}
