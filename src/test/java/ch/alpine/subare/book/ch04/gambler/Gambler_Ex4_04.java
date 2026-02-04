// code by jph
// inspired by Shangtong Zhang
package ch.alpine.subare.book.ch04.gambler;

import java.io.IOException;

import ch.alpine.subare.alg.ValueIteration;
import ch.alpine.subare.util.Policies;
import ch.alpine.subare.util.PolicyBase;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Last;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.Put;

/* package */ enum Gambler_Ex4_04 {
  ;
  static void main() throws IOException {
    GamblerModel gamblerModel = GamblerModel.createDefault();
    ValueIteration vi = new ValueIteration(gamblerModel, gamblerModel);
    Tensor record = Tensors.empty();
    for (int iters = 0; iters < 20; ++iters) {
      vi.step();
      record.append(vi.vs().values());
    }
    Tensor values = Last.of(record);
    // .untilBelow(RealScalar.of(1e-10));
    // System.out.println(values);
    Put.of(HomeDirectory.file("ex403_values"), values);
    Put.of(HomeDirectory.file("ex403_record"), record);
    PolicyBase policy = PolicyType.GREEDY.bestEquiprobable(gamblerModel, vi.vs(), null);
    Policies.print(policy, gamblerModel.states());
    Tensor greedy = Policies.flatten(policy, gamblerModel.states());
    Put.of(HomeDirectory.file("ex403_greedy"), greedy);
  }
}
