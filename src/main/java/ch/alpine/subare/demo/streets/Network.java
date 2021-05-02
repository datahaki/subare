// code by jph
package ch.alpine.subare.demo.streets;

import java.util.List;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Array;

abstract class Network {
  abstract int actions();

  abstract int streets();

  abstract List<Integer> streetsFromAction(int k);

  abstract Tensor affine();

  abstract Tensor linear();

  Tensor usage;

  final void reset() {
    usage = Array.zeros(streets());
  }

  final void feedAction(int k) {
    for (int index : streetsFromAction(k))
      usage.set(use -> use.add(RealScalar.ONE), index);
  }

  final Tensor cost() {
    return affine().add(usage.pmul(linear()));
  }
}
