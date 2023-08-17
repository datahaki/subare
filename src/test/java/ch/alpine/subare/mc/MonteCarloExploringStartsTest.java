// code by jph
package ch.alpine.subare.mc;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.api.MonteCarloInterface;
import ch.alpine.subare.api.Policy;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.subare.util.SimpleTestModel;
import ch.alpine.subare.util.SimpleTestModels;

class MonteCarloExploringStartsTest {
  @Test
  void testSimple() {
    MonteCarloInterface monteCarloInterface = SimpleTestModel.INSTANCE;
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(monteCarloInterface);
    Policy policy = PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, mces.qsa(), mces.sac());
    ExploringStarts.batch(monteCarloInterface, policy, mces);
    // DiscreteUtils.print(mces.qsa());
    SimpleTestModels._checkExact(mces.qsa());
  }
}
