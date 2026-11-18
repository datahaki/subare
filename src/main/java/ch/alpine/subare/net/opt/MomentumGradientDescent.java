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

class MomentumGradientDescent implements NetOptimizer, Serializable {
  private class Updater {
    private final ParametricLayer parametricLayer;
    private Tensor v = null;

    public Updater(ParametricLayer parametricLayer) {
      this.parametricLayer = parametricLayer;
    }

    public void update() {
      Tensor g = parametricLayer.gradient(weightDecay);
      v = Objects.isNull(v) ? g : v.multiply(beta).add(g);
      parametricLayer.increment(weightDecay, v, eta);
    }
  }

  private final List<Updater> list;
  private final boolean weightDecay;
  private final Scalar beta;
  private final Scalar eta;

  public MomentumGradientDescent(NetLayer netLayer, boolean weightDecay, Scalar eta, Scalar beta) {
    list = netLayer.stream() //
        .filter(ParametricLayer.class::isInstance) //
        .map(ParametricLayer.class::cast) //
        .map(Updater::new) //
        .collect(Collectors.toCollection(ArrayList::new));
    this.weightDecay = weightDecay;
    this.eta = eta;
    this.beta = beta;
  }

  @Override
  public void update() {
    list.forEach(Updater::update);
  }
}
