// code by jph
package ch.alpine.subare.core.util.gfx;

import java.awt.Dimension;
import java.awt.Point;

import ch.alpine.tensor.Tensor;

public interface StateRaster extends BaseRaster {
  /** @return dimension of raster */
  Dimension dimensionStateRaster();

  /** @param state
   * @return point with x, y as coordinates of state in raster,
   * or null if state does not have a position in the raster */
  Point point(Tensor state);
}
