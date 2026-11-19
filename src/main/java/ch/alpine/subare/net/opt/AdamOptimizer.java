// code by jph
package ch.alpine.subare.net.opt;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

import ch.alpine.subare.net.NetExtract;
import ch.alpine.subare.net.api.NetLayer;
import ch.alpine.subare.net.api.NetOptimizer;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.api.ScalarUnaryOperator;
import ch.alpine.tensor.itp.LinearBinaryAverage;
import ch.alpine.tensor.red.Entrywise;
import ch.alpine.tensor.sca.pow.Sqrt;

/** Adaptive Moment Estimation */
class AdamOptimizer implements NetOptimizer, Serializable {
  private class Updater {
    private static final Scalar EPS = RealScalar.of(1E-8);
    private static final ScalarUnaryOperator SUO = s -> Sqrt.FUNCTION.apply(s).add(EPS);
    private final NetLayer parametricLayer;
    Tensor m;
    Tensor v;
    Scalar beta1_t = RealScalar.ONE;
    Scalar beta2_t = RealScalar.ONE;

    public Updater(NetLayer parametricLayer) {
      this.parametricLayer = parametricLayer;
    }

    public void update() {
      Tensor g = parametricLayer.gradient(weightDecay);
      // ---
      m = Objects.isNull(m) ? g : LinearBinaryAverage.INSTANCE.split(g, m, beta1);
      Tensor q = Entrywise.mul().apply(g, g);
      v = Objects.isNull(v) ? q : LinearBinaryAverage.INSTANCE.split(q, v, beta2);
      beta1_t = beta1_t.multiply(beta1);
      Tensor m_h = m.divide(RealScalar.ONE.subtract(beta1_t));
      beta2_t = beta2_t.multiply(beta2);
      Tensor v_h = v.divide(RealScalar.ONE.subtract(beta2_t));
      Tensor delta = Entrywise.div().apply(m_h, v_h.maps(SUO));
      // ---
      parametricLayer.increment(weightDecay, delta, initialized ? eta : eta.zero());
    }
  }

  private final List<Updater> list;
  private final boolean weightDecay;
  private final Scalar eta;
  private final Scalar beta1;
  private final Scalar beta2;
  private boolean initialized = false;

  public AdamOptimizer(NetLayer netLayer, boolean weightDecay, Scalar eta, Scalar beta1, Scalar beta2) {
    list = NetExtract.flatten(netLayer) //
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
