// code by jph
package ch.alpine.subare.ch08.maze;

import ch.alpine.subare.core.alg.ActionValueIterations;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.io.ResourceData;

public enum DynamazeHelper {
  ;
  private static final Tensor STARTS_5 = Tensors.matrixInt(new int[][] { //
      { 31, 15 }, { 9, 15 }, { 18, 12 } });

  /** @param name, for instance "maze2"
   * @return */
  public static Dynamaze original(String name) {
    return fromImage(load(name));
  }

  public static Dynamaze create5(int starts) {
    Tensor image = load("maze5");
    for (int count = 0; count < starts; ++count) {
      Tensor vec = STARTS_5.get(count);
      image.set(Dynamaze.GREEN, //
          Scalars.intValueExact(vec.Get(0)), //
          Scalars.intValueExact(vec.Get(1)));
    }
    return fromImage(image);
  }

  private static Dynamaze fromImage(Tensor image) {
    return new Dynamaze(image.unmodifiable());
  }

  /* package */ static Tensor load(String name) {
    return ResourceData.of("/ch08/" + name + ".png");
  }

  static DiscreteQsa getOptimalQsa(Dynamaze dynamaze) {
    return ActionValueIterations.solve(dynamaze, RealScalar.of(.0000001));
  }
}
