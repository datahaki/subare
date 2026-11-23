// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;
import java.util.Objects;
import java.util.random.RandomGenerator;

import ch.alpine.subare.net.TrainableNetLayer;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Slash;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.VectorQ;
import ch.alpine.tensor.ext.Int;
import ch.alpine.tensor.nrm.Vector2NormSquared;
import ch.alpine.tensor.red.Entrywise;
import ch.alpine.tensor.red.Times;
import ch.alpine.tensor.red.Total;
import ch.alpine.tensor.sca.pow.Sqrt;

public class RMSLayer implements TrainableNetLayer, Serializable {
  private static final Scalar EPS = RealScalar.of(1E-12);
  // ---
  private Tensor gamma = null;
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
      gamma = VectorQ.require(X.get(0)).maps(Scalar::one);
    rmsx = Slash.of(X, RMSLayer::rms);
    xhat = Times.of(rmsx.maps(Scalar::reciprocal), X);
    return Slash.of(xhat, Entrywise.mul().operator(gamma));
  }

  private static Scalar rms(Tensor x) {
    return Sqrt.FUNCTION.apply(Vector2NormSquared.of(x).divide(RealScalar.of(x.length())).add(EPS));
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    gG = Entrywise.mul().apply(gradOutput, xhat);
    Scalar d = d();
    Tensor u = Entrywise.mul().apply(gradOutput, gamma);
    return u.subtract(xhat.multiply((Scalar) u.dot(xhat).divide(d))).divide((Scalar) rmsx);
  }

  @Override
  public Tensor backBatch(Tensor gradOutput) {
    gG = Total.of(Entrywise.mul().apply(gradOutput, xhat));
    Scalar d = d();
    Int myI = new Int();
    return Slash.of(gradOutput, grad -> {
      Tensor u = Entrywise.mul().apply(grad, gamma);
      int i = myI.getAndIncrement();
      Tensor xhati = xhat.get(i);
      return u.subtract(xhati.multiply((Scalar) u.dot(xhati).divide(d))).divide(rmsx.Get(i));
    });
  }

  @Override
  public void initialize(RandomGenerator randomGenerator) {
    // gamma is all ones
  }

  @Override
  public Tensor parameters() {
    return gamma;
  }

  @Override
  public Tensor gradient(boolean decayInLoss) {
    return gG;
  }

  @Override
  public void step(boolean decayInLoss, Tensor delta, Scalar eta) {
    gamma = gamma.add(delta.multiply(eta.negate()));
  }

  private Scalar d() {
    return RealScalar.of(gamma.length());
  }

  @Override
  public String toString() {
    return "RMSLayer";
  }
}
