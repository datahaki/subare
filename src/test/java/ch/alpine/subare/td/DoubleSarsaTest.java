// code by jph
package ch.alpine.subare.td;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.api.MonteCarloInterface;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.util.ConstantLearningRate;
import ch.alpine.subare.util.DefaultLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.LearningRate;
import ch.alpine.subare.util.PolicyBase;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.subare.util.SimpleTestModel;
import ch.alpine.subare.util.SimpleTestModels;
import ch.alpine.tensor.RealScalar;

class DoubleSarsaTest {
  @Test
  void testExact() {
    for (SarsaType sarsaType : SarsaType.values()) {
      MonteCarloInterface monteCarloInterface = SimpleTestModel.INSTANCE;
      LearningRate learningRate = DefaultLearningRate.of(RealScalar.of(2.3), RealScalar.of(0.6));
      DiscreteQsa qsa = DiscreteQsa.build(monteCarloInterface, RealScalar.ZERO);
      DiscreteQsa qsa1 = DiscreteQsa.build(monteCarloInterface, RealScalar.ZERO);
      DiscreteQsa qsa2 = DiscreteQsa.build(monteCarloInterface, RealScalar.ZERO);
      StateActionCounter sac = new DiscreteStateActionCounter();
      StateActionCounter sac1 = new DiscreteStateActionCounter();
      StateActionCounter sac2 = new DiscreteStateActionCounter();
      PolicyBase policy = PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa, sac);
      PolicyBase policy1 = PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa1, sac1);
      PolicyBase policy2 = PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa2, sac2);
      DoubleSarsa doubleSarsa = sarsaType.doubleSarsa(monteCarloInterface, learningRate, qsa1, qsa2, sac1, sac2, policy1, policy2);
      ExploringStarts.batch(monteCarloInterface, policy, 2, doubleSarsa); // nstep > 1 required
      // DiscreteUtils.print(doubleSarsa.qsa());
      // TODO SUBARE investigate why this results in numeric precision
      SimpleTestModels._checkExactNumeric(doubleSarsa.qsa());
    }
  }

  @Test
  void testExact2() {
    for (SarsaType sarsaType : SarsaType.values()) {
      MonteCarloInterface monteCarloInterface = SimpleTestModel.INSTANCE;
      LearningRate learningRate = ConstantLearningRate.one();
      DiscreteQsa qsa = DiscreteQsa.build(monteCarloInterface, RealScalar.ZERO);
      DiscreteQsa qsa1 = DiscreteQsa.build(monteCarloInterface, RealScalar.ZERO);
      DiscreteQsa qsa2 = DiscreteQsa.build(monteCarloInterface, RealScalar.ZERO);
      StateActionCounter sac = new DiscreteStateActionCounter();
      StateActionCounter sac1 = new DiscreteStateActionCounter();
      StateActionCounter sac2 = new DiscreteStateActionCounter();
      PolicyBase policy = PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa, sac);
      PolicyBase policy1 = PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa1, sac1);
      PolicyBase policy2 = PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa2, sac2);
      DoubleSarsa doubleSarsa = sarsaType.doubleSarsa(monteCarloInterface, learningRate, qsa1, qsa2, sac1, sac2, policy1, policy2);
      ExploringStarts.batch(monteCarloInterface, policy, 2, doubleSarsa); // nstep > 1 required
      // DiscreteUtils.print(doubleSarsa.qsa());
      SimpleTestModels._checkExact(doubleSarsa.qsa());
    }
  }
}
