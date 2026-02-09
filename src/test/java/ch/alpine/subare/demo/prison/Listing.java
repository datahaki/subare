// code by jph
package ch.alpine.subare.demo.prison;

import java.util.List;
import java.util.function.Supplier;

import ch.alpine.subare.book.ch02.Agent;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.sca.Round;

/* package */ enum Listing {
  ;
  static void main() {
    List<Supplier<Agent>> list = AgentSupplier.mixed;
    Tensor matrix = AllPairs.performance(list, 20, 100);
    final int size = matrix.length();
    for (int i1 = 0; i1 < size; ++i1) {
      for (int i2 = i1; i2 < size; ++i2) {
        Agent a1 = list.get(i1).get();
        Agent a2 = list.get(i2).get();
        System.out.println("---");
        System.out.printf("%s %s%n", //
            a1.getAbsDesc(), matrix.Get(i1, i2).maps(Round._3));
        System.out.printf("%s %s%n", //
            a2.getAbsDesc(), matrix.Get(i2, i1).maps(Round._3));
      }
    }
    System.out.println("done.");
    int asd = Integer.parseInt("000123");
    System.out.println(asd);
  }
}
