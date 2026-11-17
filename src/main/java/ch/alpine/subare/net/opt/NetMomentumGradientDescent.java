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
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

public class NetMomentumGradientDescent implements NetOptimizer, Serializable {
  private static interface Updater {
    void update(Scalar beta, Scalar eta);
  }

  private static class LLUpdater implements Updater {
    private final LinearLayer linearLayer;
    Tensor v = null;

    public LLUpdater(LinearLayer linearLayer) {
      this.linearLayer = linearLayer;
    }

    @Override
    public void update(Scalar eta, Scalar beta) {
      Tensor g = linearLayer.gradient();
      v = Objects.isNull(v) //
          ? g
          : v.multiply(beta).add(g);
      linearLayer.setWb(linearLayer.getWb().add(v.multiply(eta.negate())));
    }
  }

  private static Optional<Updater> buildMap(NetLayer netLayer) {
    if (netLayer instanceof LinearLayer linearLayer)
      return Optional.of(new LLUpdater(linearLayer));
    return Optional.empty();
  }

  private final Scalar beta;
  private final Scalar eta;
  private final List<Updater> list;

  public NetMomentumGradientDescent(NetLayer netLayer, Scalar eta, Scalar beta) {
    list = netLayer.stream() //
        .map(NetMomentumGradientDescent::buildMap) //
        .flatMap(Optional::stream) //
        .collect(Collectors.toCollection(ArrayList::new));
    this.eta = eta;
    this.beta = beta;
  }

  @Override
  public void update() {
    list.forEach(updater -> updater.update(eta, beta));
  }
}
