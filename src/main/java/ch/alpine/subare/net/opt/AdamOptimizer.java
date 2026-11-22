// code by jph
package ch.alpine.subare.net.opt;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.NetLayers;
import ch.alpine.subare.net.NetOptimizable;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.api.ScalarUnaryOperator;
import ch.alpine.tensor.itp.LinearBinaryAverage;
import ch.alpine.tensor.red.Entrywise;
import ch.alpine.tensor.sca.pow.Sqrt;

/** Adaptive Moment Estimation */
// TODO should probably not be serializable!!!
class AdamOptimizer implements NetOptimizer, Serializable {
  private class Updater {
    private static final Scalar EPS = RealScalar.of(1E-12);
    private static final ScalarUnaryOperator SUO = s -> Sqrt.FUNCTION.apply(s).add(EPS);
    private final NetOptimizable netOptimizable;
    Tensor m;
    Tensor v;
    Scalar beta1_t = RealScalar.ONE;
    Scalar beta2_t = RealScalar.ONE;

    public Updater(NetOptimizable netOptimizable) {
      this.netOptimizable = netOptimizable;
    }

    public void update() {
      Tensor g = netOptimizable.gradient(weightDecay);
      // ---
      m = Objects.isNull(m) ? g : LinearBinaryAverage.INSTANCE.split(g, m, beta1);
      Tensor q = Entrywise.mul().apply(g, g);
      v = Objects.isNull(v) ? q : LinearBinaryAverage.INSTANCE.split(q, v, beta2);
      // Instead of multiplying beta1_t and beta2_t on every step (which can accumulate floating-point
      // drift over long runs), consider tracking long step = 0
      // and computing beta1.pow(step) / beta2.pow(step) if your Scalar type supports .pow(...).
      beta1_t = beta1_t.multiply(beta1);
      Tensor m_h = m.divide(RealScalar.ONE.subtract(beta1_t));
      beta2_t = beta2_t.multiply(beta2);
      Tensor v_h = v.divide(RealScalar.ONE.subtract(beta2_t));
      Tensor delta = Entrywise.div().apply(m_h, v_h.maps(SUO));
      // ---
      netOptimizable.step(weightDecay, delta, eta);
    }
  }

  private final List<Updater> list;
  private final boolean weightDecay;
  private final Scalar eta;
  private final Scalar beta1;
  private final Scalar beta2;
  private boolean initialized = false;

  public AdamOptimizer(NetLayer netLayer, boolean weightDecay, Scalar eta, Scalar beta1, Scalar beta2) {
    list = NetLayers.trainableLayers(netLayer) //
        .map(Updater::new) //
        .collect(Collectors.toCollection(ArrayList::new));
    this.weightDecay = weightDecay;
    this.eta = eta;
    this.beta1 = beta1;
    this.beta2 = beta2;
  }

  @Override
  public void update() {
    list.forEach(Updater::update);
    initialized = true;
  }

  @Override
  public boolean weightDecay() {
    return weightDecay;
  }
}
