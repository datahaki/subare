// code by jph
package ch.alpine.subare.net.opt;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.lay.ParametricLayer;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

public class NetMomentumGradientDescent implements NetOptimizer, Serializable {
  private static class Updater {
    private final ParametricLayer parametricLayer;
    private Tensor v = null;

    public Updater(ParametricLayer parametricLayer) {
      this.parametricLayer = parametricLayer;
    }

    public void update(Scalar eta, Scalar beta) {
      Tensor g = parametricLayer.gradient();
      v = Objects.isNull(v) ? g : v.multiply(beta).add(g);
      parametricLayer.increment(v.multiply(eta.negate()));
    }
  }

  private final Scalar beta;
  private final Scalar eta;
  private final List<Updater> list;

  public NetMomentumGradientDescent(NetLayer netLayer, Scalar eta, Scalar beta) {
    list = netLayer.stream() //
        .filter(ParametricLayer.class::isInstance) //
        .map(ParametricLayer.class::cast) //
        .map(Updater::new) //
        .collect(Collectors.toCollection(ArrayList::new));
    this.eta = eta;
    this.beta = beta;
  }

  @Override
  public void update() {
    list.forEach(updater -> updater.update(eta, beta));
  }
}
