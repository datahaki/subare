// code by jph
package ch.alpine.subare.net.run;

import java.util.stream.Stream;

import ch.alpine.subare.net.lay.LinearLayer;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.lay.ParametricLayer;
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

  public Scalar reg_loss() {
    return netLayer.stream().flatMap(this::reg_loss).reduce(Scalar::add).orElse(RealScalar.ZERO);
  }

  private Stream<Scalar> reg_loss(NetLayer netLayer) {
    if (netLayer instanceof LinearLayer linearLayer)
      return Stream.of(linearLayer.reg_loss());
    return Stream.empty();
  }
}
