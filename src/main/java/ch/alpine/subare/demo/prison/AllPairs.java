// code by jph
package ch.alpine.subare.demo.prison;

import java.util.List;
import java.util.function.Supplier;

import ch.alpine.subare.ch02.Agent;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.ext.Integers;
import ch.alpine.tensor.red.Mean;
import ch.alpine.tensor.sca.Chop;

/* package */ enum AllPairs {
  ;
  static Tensor performance(List<Supplier<Agent>> list, int runs, int epochs) {
    final int size = list.size();
    Tensor matrix = Array.zeros(size, size);
    for (int i1 = 0; i1 < size; ++i1) {
      for (int i2 = i1; i2 < size; ++i2) {
        Tensor table = Tensors.empty();
        for (int run = 0; run < runs; ++run) {
          Agent a1 = list.get(i1).get();
          Agent a2 = list.get(i2).get();
          table.append(Training.train(a1, a2, epochs));
        }
        Integers.requireEquals(table.length(), runs);
        Tensor mean = Mean.of(table);
        Chop.NONE.requireAllZero(matrix.Get(i1, i2));
        Chop.NONE.requireAllZero(matrix.Get(i2, i1));
        matrix.set(mean.Get(0), i1, i2);
        matrix.set(mean.Get(1), i2, i1);
      }
    }
    return matrix;
  }
}
