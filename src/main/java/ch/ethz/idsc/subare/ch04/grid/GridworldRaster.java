// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import java.awt.Dimension;
import java.awt.Point;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRaster;
import ch.ethz.idsc.subare.core.util.gfx.StateRaster;
import ch.ethz.idsc.subare.core.util.gfx.StateRasters;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;

class GridworldRaster implements StateRaster, StateActionRaster {
  private final Gridworld gridworld;
  private final Index indexActions;

  public GridworldRaster(Gridworld gridworld) {
    this.gridworld = gridworld;
    indexActions = Index.build(gridworld.actions(null));
  }

  @Override
  public DiscreteModel discreteModel() {
    return gridworld;
  }

  @Override
  public Dimension dimensionStateRaster() {
    return new Dimension(Gridworld.NX, Gridworld.NY);
  }

  @Override
  public Point point(Tensor state) {
    return StateRasters.canonicPoint(state);
  }

  @Override
  public Dimension dimensionStateActionRaster() {
    return new Dimension((Gridworld.NX + 1) * 4 - 1, Gridworld.NY);
  }

  @Override
  public Point point(Tensor state, Tensor action) {
    int sx = Scalars.intValueExact(state.Get(0));
    int sy = Scalars.intValueExact(state.Get(1));
    int a = indexActions.of(action);
    return new Point(sx + (Gridworld.NX + 1) * a, sy);
  }

  @Override
  public Scalar scaleLoss() {
    return RealScalar.of(1);
  }

  @Override
  public Scalar scaleQdelta() {
    return RealScalar.of(15);
  }

  @Override
  public int joinAlongDimension() {
    return 0;
  }

  @Override
  public int magnify() {
    return 7;
  }
}
