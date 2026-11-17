// code by jph
package ch.alpine.subare.net.opt;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.lay.ParametricLayer;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

/** simple gradient descent */
class SimpleGradientDescent implements NetOptimizer, Serializable {
  private final List<ParametricLayer> list;
  private final Scalar eta;

  public SimpleGradientDescent(NetLayer netLayer, Scalar eta) {
    list = netLayer.stream() //
        .filter(ParametricLayer.class::isInstance) //
        .map(ParametricLayer.class::cast) //
        .collect(Collectors.toCollection(ArrayList::new));
    this.eta = eta;
  }

  private void update(ParametricLayer linearLayer) {
    Tensor g = linearLayer.gradient();
    linearLayer.increment(g.multiply(eta.negate()));
  }

  @Override
  public void update() {
    list.forEach(updater -> update(updater));
  }
}
