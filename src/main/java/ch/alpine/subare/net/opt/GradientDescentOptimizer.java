// code by jph
package ch.alpine.subare.net.opt;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import ch.alpine.subare.net.NetExtract;
import ch.alpine.subare.net.api.NetLayer;
import ch.alpine.subare.net.api.NetOptimizer;
import ch.alpine.tensor.Scalar;

/** simple gradient descent */
class GradientDescentOptimizer implements NetOptimizer, Serializable {
  private final List<NetLayer> list;
  private final Scalar eta;

  public GradientDescentOptimizer(NetLayer netLayer, Scalar eta) {
    list = NetExtract.flatten(netLayer) //
        .collect(Collectors.toCollection(ArrayList::new));
    this.eta = eta;
  }

  private void update(NetLayer linearLayer) {
    linearLayer.increment(true, linearLayer.gradient(true), eta);
  }

  @Override
  public void update() {
    list.forEach(updater -> update(updater));
  }

  @Override
  public boolean weightDecay() {
    return true;
  }
}
