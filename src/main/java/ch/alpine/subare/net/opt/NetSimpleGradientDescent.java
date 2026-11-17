// code by jph
package ch.alpine.subare.net.opt;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

import ch.alpine.subare.net.lay.LinearLayer;
import ch.alpine.subare.net.lay.LinearLayer.AffineTuple;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.tensor.Scalar;

/** simple gradient descent */
public class NetSimpleGradientDescent implements NetOptimizer, Serializable {
  private static interface Updater {
    void update(Scalar eta);
  }

  private static class LLUpdater implements Updater {
    private final LinearLayer linearLayer;

    public LLUpdater(LinearLayer linearLayer) {
      this.linearLayer = linearLayer;
    }

    @Override
    public void update(Scalar eta) {
      AffineTuple affineTuple = linearLayer.getWb();
      linearLayer.setWb( //
          affineTuple.W().add(linearLayer.gradient_W().multiply(eta.negate())), //
          affineTuple.b().add(linearLayer.gradient_b().multiply(eta.negate())));
    }
  }

  private static Optional<Updater> buildMap(NetLayer netLayer) {
    if (netLayer instanceof LinearLayer linearLayer)
      return Optional.of(new LLUpdater(linearLayer));
    return Optional.empty();
  }

  private final Scalar eta;
  private final List<Updater> list;

  public NetSimpleGradientDescent(NetLayer netLayer, Scalar eta) {
    this.eta = eta;
    list = netLayer.stream() //
        .map(NetSimpleGradientDescent::buildMap) //
        .flatMap(Optional::stream) //
        .collect(Collectors.toCollection(ArrayList::new));
  }

  @Override
  public void update() {
    list.forEach(updater -> updater.update(eta));
  }
}
