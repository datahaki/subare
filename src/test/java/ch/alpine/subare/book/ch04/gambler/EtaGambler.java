// code by jph
package ch.alpine.subare.book.ch04.gambler;

import java.util.List;

import ch.alpine.subare.alg.OnPolicyStateDistribution;
import ch.alpine.subare.api.Policy;
import ch.alpine.subare.util.DiscreteVs;
import ch.alpine.tensor.RationalScalar;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.ArrayPad;
import ch.alpine.tensor.alg.ConstantArray;
import ch.alpine.tensor.nrm.NormalizeTotal;
import ch.alpine.tensor.red.Total;
import ch.alpine.tensor.sca.Sign;

/* package */ enum EtaGambler {
  ;
  static void main() {
    GamblerModel gamblerModel = new GamblerModel(10, RationalScalar.of(4, 10));
    // Policy policy = EquiprobablePolicy.create(gambler);
    Policy policy = GamblerHelper.getOptimalPolicy(gamblerModel);
    OnPolicyStateDistribution opsd = new OnPolicyStateDistribution(gamblerModel, gamblerModel, policy);
    Tensor values = //
        ArrayPad.of(ConstantArray.of(RealScalar.ONE, 9), List.of(1), List.of(1));
    values.map(Sign::requirePositiveOrZero);
    values = NormalizeTotal.FUNCTION.apply(values);
    Scalar scalar = Total.ofVector(values);
    System.out.println("sum=" + scalar);
    DiscreteVs vs = DiscreteVs.build(gamblerModel.states(), values);
    for (int count = 0; count < 10; ++count) {
      vs = opsd.iterate(vs);
      System.out.println(vs.values());
      System.out.println("total=" + Total.of(vs.values()));
    }
  }
}
