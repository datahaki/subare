// code by jph
package ch.alpine.subare.net.run;

import java.util.stream.Stream;

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
    return netLayer.stream() //
        .filter(ParametricLayer.class::isInstance) //
        .map(ParametricLayer.class::cast) //
        .map(ParametricLayer::reg_loss) //
        .reduce(Scalar::add).orElse(RealScalar.ZERO);
  }
}
