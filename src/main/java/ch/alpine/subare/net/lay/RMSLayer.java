// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;
import java.util.Objects;

import ch.alpine.subare.net.api.NetLayer;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.io.MathematicaFormat;
import ch.alpine.tensor.nrm.Vector2NormSquared;
import ch.alpine.tensor.red.Entrywise;
import ch.alpine.tensor.red.Times;
import ch.alpine.tensor.red.Total;
import ch.alpine.tensor.sca.pow.Sqrt;

public class RMSLayer implements NetLayer, Serializable {
  private static final Scalar EPS = RealScalar.of(1E-12);
  // ---
  public Tensor gamma = null;
  private transient Tensor rmsx; // scalar (or vector for batches)
  private transient Tensor xhat; // vector (or matrix for batches)
  private transient Tensor gG; // vector with same length as gamma

  @Override
  public Tensor forward(Tensor x) {
    if (Objects.isNull(gamma))
      gamma = x.maps(Scalar::one);
    rmsx = rms(x);
    xhat = x.divide((Scalar) rmsx);
    return Entrywise.mul().apply(xhat, gamma);
  }

  @Override
  public Tensor forwardBatch(Tensor X) {
    if (Objects.isNull(gamma))
      gamma = X.get(0).maps(Scalar::one);
    rmsx = Tensor.of(X.stream().map(RMSLayer::rms));
    xhat = Times.of(rmsx.maps(Scalar::reciprocal), X);
    return Tensor.of(xhat.stream().map(row -> Entrywise.mul().apply(row, gamma)));
  }

  private static Scalar rms(Tensor x) {
    return Sqrt.FUNCTION.apply(Vector2NormSquared.of(x).divide(RealScalar.of(x.length())).add(EPS));
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    gG = Entrywise.mul().apply(gradOutput, xhat);
    Tensor u = Entrywise.mul().apply(gradOutput, gamma);
    return u.subtract(xhat.multiply((Scalar) u.dot(xhat).divide(d()))).divide((Scalar) rmsx);
  }

  @Override
  public Tensor backBatch(Tensor gradOutput) {
    gG = Total.of(Entrywise.mul().apply(gradOutput, xhat));
    Tensor U = Tensors.reserve(gradOutput.length());
    // TODO possibly improve code below?
    for (int i = 0; i < gradOutput.length(); ++i) {
      Tensor u = Entrywise.mul().apply(gradOutput.get(i), gamma);
      U.append(u.subtract(xhat.get(i).multiply((Scalar) u.dot(xhat.get(i)).divide(d()))).divide(rmsx.Get(i)));
    }
    return U;
  }

  @Override
  public Tensor gradient(boolean decayInLoss) {
    return gG;
  }

  @Override
  public void increment(boolean decayInLoss, Tensor delta, Scalar eta) {
    gamma = gamma.add(delta.multiply(eta.negate()));
  }

  private Scalar d() {
    return RealScalar.of(gamma.length());
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("RMSLayer", gamma);
  }
}
