// code by jph
package ch.alpine.subare.net.opt;

import java.util.Objects;

import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.api.ScalarUnaryOperator;
import ch.alpine.tensor.itp.LinearBinaryAverage;
import ch.alpine.tensor.red.Entrywise;
import ch.alpine.tensor.sca.pow.Power;
import ch.alpine.tensor.sca.pow.Sqrt;

/** Adaptive Moment Estimation */
class AdamOptimizer extends AbstractNetOptimizer {
  private class Updater implements Runnable {
    private static final Scalar EPS = RealScalar.of(1E-12);
    private static final ScalarUnaryOperator SUO = s -> Sqrt.FUNCTION.apply(s).add(EPS);
    final NetOptimizable netOptimizable;
    Tensor m;
    Tensor v;

    public Updater(NetOptimizable netOptimizable) {
      this.netOptimizable = netOptimizable;
    }

    @Override
    public void run() {
      Tensor g = netOptimizable.gradient(weightDecay);
      // ---
      m = Objects.isNull(m) ? g : LinearBinaryAverage.INSTANCE.split(g, m, beta1);
      Tensor q = Entrywise.mul().apply(g, g);
      v = Objects.isNull(v) ? q : LinearBinaryAverage.INSTANCE.split(q, v, beta2);
      Tensor m_h = m.divide(RealScalar.ONE.subtract(Power.of(beta1, count + 1)));
      Tensor v_h = v.divide(RealScalar.ONE.subtract(Power.of(beta2, count + 1)));
      Tensor delta = Entrywise.div().apply(m_h, v_h.maps(SUO));
      // ---
      netOptimizable.step(weightDecay, delta, eta());
    }
  }

  private final Scalar beta1;
  private final Scalar beta2;

  public AdamOptimizer(NetLayer netLayer, boolean weightDecay, Scalar eta, Scalar beta1, Scalar beta2) {
    super(netLayer, weightDecay, eta);
    this.beta1 = beta1;
    this.beta2 = beta2;
  }

  @Override
  Runnable create(NetOptimizable netOptimizable) {
    return new Updater(netOptimizable);
  }
}
