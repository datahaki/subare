// code by jph
package ch.alpine.subare.demo.streets;

import java.util.List;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;

/** example network by Julian
 * three possible actions */
class BridgeNetwork extends Network {
  private Tensor affine = Tensors.vectorInt(-3, 0, 0, -3, 0).unmodifiable();
  private Tensor linear = Tensors.vectorInt(0, -1, -1, 0, 0).unmodifiable();

  @Override
  int actions() {
    return 3;
  }

  @Override
  int streets() {
    return 5;
  }

  @Override
  List<Integer> streetsFromAction(int k) {
    return switch (k) {
    case 0 -> List.of(0, 1);
    case 1 -> List.of(2, 3);
    case 2 -> List.of(2, 4, 1);
    default -> throw new IllegalArgumentException();
    };
  }

  @Override
  Tensor affine() {
    return affine;
  }

  @Override
  Tensor linear() {
    return linear;
  }
}
