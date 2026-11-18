// code by jph
package ch.alpine.subare.net;

import java.util.stream.Stream;

import ch.alpine.subare.net.api.NetLayer;
import ch.alpine.subare.net.api.ParametricLayer;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.alg.Flatten;

public record NetExtract(NetLayer netLayer) {
  public Stream<Scalar> all() {
    return netLayer.stream() //
        .filter(ParametricLayer.class::isInstance) //
        .map(ParametricLayer.class::cast) //
        .map(ParametricLayer::parameters) //
        .flatMap(Flatten::scalars);
  }

  public Scalar reg_loss(boolean weightDecay) {
    return netLayer.stream() //
        .filter(ParametricLayer.class::isInstance) //
        .map(ParametricLayer.class::cast) //
        .map(p -> p.reg_loss(weightDecay)) //
        .reduce(Scalar::add) //
        .orElse(RealScalar.ZERO);
  }
}
