// code by jph
package ch.alpine.subare.book.ch08.maze;

import java.awt.Dimension;
import java.util.List;

import ch.alpine.subare.api.DiscreteModel;
import ch.alpine.subare.util.gfx.StateRaster;
import ch.alpine.subare.util.gfx.StateRasters;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Dimensions;

class DynamazeRaster implements StateRaster {
  private final Dynamaze dynamaze;

  public DynamazeRaster(Dynamaze dynamaze) {
    this.dynamaze = dynamaze;
  }

  @Override
  public DiscreteModel discreteModel() {
    return dynamaze;
  }

  @Override
  public Dimension dimensionStateRaster() {
    List<Integer> dimensions = Dimensions.of(dynamaze.image());
    return new Dimension(dimensions.get(1), dimensions.get(0));
  }

  @Override
  public List<Integer> point(Tensor state) {
    return StateRasters.canonicPoint(state);
  }

  @Override
  public Scalar scaleLoss() {
    return RealScalar.of(50.);
  }

  @Override
  public Scalar scaleQdelta() {
    return RealScalar.of(50.);
  }

  @Override
  public int joinAlongDimension() {
    return 1;
  }

  @Override
  public int magnify() {
    return 4;
  }
}
