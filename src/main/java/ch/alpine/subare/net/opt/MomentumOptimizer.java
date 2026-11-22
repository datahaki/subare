// code by jph
package ch.alpine.subare.net.opt;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

import ch.alpine.subare.net.NetOptimizable;
import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.NetLayers;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

// TODO refactor so that Adam and Moment have a common base class! 
class MomentumOptimizer implements NetOptimizer, Serializable {
  private class Updater {
    private final NetOptimizable parametricLayer;
    private Tensor v = null;

    public Updater(NetOptimizable parametricLayer) {
      this.parametricLayer = parametricLayer;
    }

    public void update() {
      Tensor g = parametricLayer.gradient(weightDecay);
      v = Objects.isNull(v) ? g : v.multiply(beta).add(g);
      parametricLayer.step(weightDecay, v, initialized ? eta : eta.zero());
    }
  }

  private final List<Updater> list;
  private final boolean weightDecay;
  private final Scalar beta;
  private final Scalar eta;
  private boolean initialized = false;

  public MomentumOptimizer(NetLayer netLayer, boolean weightDecay, Scalar eta, Scalar beta) {
    list = NetLayers.trainableLayers(netLayer) //
        .map(Updater::new) //
        .collect(Collectors.toCollection(ArrayList::new));
    this.weightDecay = weightDecay;
    this.eta = eta;
    this.beta = beta;
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
