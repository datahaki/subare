// code by jph
package ch.alpine.subare.ch05.wireloop;

import java.awt.Dimension;
import java.awt.Point;
import java.util.List;

import ch.alpine.subare.core.DiscreteModel;
import ch.alpine.subare.core.util.gfx.StateRaster;
import ch.alpine.subare.core.util.gfx.StateRasters;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Dimensions;

class WireloopRaster implements StateRaster {
  private final Wireloop wireloop;

  public WireloopRaster(Wireloop wireloop) {
    this.wireloop = wireloop;
  }

  @Override
  public DiscreteModel discreteModel() {
    return wireloop;
  }

  @Override
  public Dimension dimensionStateRaster() {
    List<Integer> dimensions = Dimensions.of(wireloop.image());
    return new Dimension(dimensions.get(1), dimensions.get(0));
  }

  @Override
  public Point point(Tensor state) {
    return StateRasters.canonicPoint(state);
  }

  @Override
  public Scalar scaleLoss() {
    return RealScalar.of(100.);
  }

  @Override
  public Scalar scaleQdelta() {
    return RealScalar.ONE;
  }

  @Override
  public int joinAlongDimension() {
    return 1;
  }

  @Override
  public int magnify() {
    return 2;
  }
}
