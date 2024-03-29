// code by jph
package ch.alpine.subare.util.gfx;

import java.awt.Dimension;
import java.awt.Point;

import ch.alpine.tensor.Tensor;

public interface StateActionRaster extends BaseRaster {
  /** @return dimension of raster */
  Dimension dimensionStateActionRaster();

  /** @param state
   * @param action
   * @return point with x, y as coordinates of state-action pair in raster,
   * or null if state-action pair does not have a position in the raster */
  Point point(Tensor state, Tensor action);
}
