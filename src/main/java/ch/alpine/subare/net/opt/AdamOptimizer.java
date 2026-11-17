// code by jph
package ch.alpine.subare.net.opt;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.stream.Collectors;

import ch.alpine.subare.net.lay.LinearLayer;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.api.ScalarUnaryOperator;
import ch.alpine.tensor.itp.LinearBinaryAverage;
import ch.alpine.tensor.red.Entrywise;
import ch.alpine.tensor.sca.pow.Sqrt;

public class AdamOptimizer implements NetOptimizer, Serializable {
  private static interface Updater {
    void update(Scalar beta1, Scalar beta2, Scalar eta);
  }

  private static class LLUpdater implements Updater {
    static final Scalar EPS = RealScalar.of(1E-8);
    static final ScalarUnaryOperator SUO = s -> Sqrt.FUNCTION.apply(s).add(EPS);
    private final LinearLayer linearLayer;
    // velocity
    Tensor m;
    Tensor v;
    Scalar beta1_t = RealScalar.ONE;
    Scalar beta2_t = RealScalar.ONE;

    public LLUpdater(LinearLayer linearLayer) {
      this.linearLayer = linearLayer;
    }

    @Override
    public void update(Scalar eta, Scalar beta1, Scalar beta2) {
      Tensor g = linearLayer.gradient();
      m = Objects.isNull(m) ? g : LinearBinaryAverage.INSTANCE.split(g, m, beta1);
      Tensor q = Entrywise.mul().apply(g, g);
      v = Objects.isNull(v) ? q : LinearBinaryAverage.INSTANCE.split(q, v, beta2);
      beta1_t = beta1_t.multiply(beta1);
      Tensor m_h = m.divide(RealScalar.ONE.subtract(beta1_t));
      beta2_t = beta2_t.multiply(beta2);
      Tensor v_h = v.divide(RealScalar.ONE.subtract(beta2_t));
      Tensor r = Entrywise.div().apply(m_h, v_h.maps(SUO)).multiply(eta.negate());
      Tensor Wb = linearLayer.getWb();
      linearLayer.setWb(Wb.add(r));
    }
  }

  private static Optional<Updater> buildMap(NetLayer netLayer) {
    if (netLayer instanceof LinearLayer linearLayer)
      return Optional.of(new LLUpdater(linearLayer));
    return Optional.empty();
  }

  private final Scalar eta;
  private final Scalar beta1;
  private final Scalar beta2;
  private final List<Updater> list;

  public AdamOptimizer(NetLayer netLayer, Scalar eta, Scalar beta1, Scalar beta2) {
    list = netLayer.stream() //
        .map(AdamOptimizer::buildMap) //
        .flatMap(Optional::stream) //
        .collect(Collectors.toCollection(ArrayList::new));
    this.eta = eta;
    this.beta1 = beta1;
    this.beta2 = beta2;
  }

  @Override
  public void update() {
    list.forEach(updater -> updater.update(eta, beta1, beta2));
  }
}
