// code by jph
package ch.alpine.subare.book.ch08.maze;

import java.nio.file.Path;

import ch.alpine.subare.alg.ActionValueIterations;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.Unprotect;
import ch.alpine.tensor.io.Import;

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
    Path file = Unprotect.path("/ch08/" + name + ".png");
    try {
      return Import.of(file);
    } catch (Exception exception) {
      throw new RuntimeException(exception);
    }
  }

  static DiscreteQsa getOptimalQsa(Dynamaze dynamaze) {
    return ActionValueIterations.solve(dynamaze, RealScalar.of(.0000001));
  }
}
