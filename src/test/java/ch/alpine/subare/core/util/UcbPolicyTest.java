// code by jph
package ch.alpine.subare.core.util;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.core.StateActionCounter;
import ch.alpine.subare.core.adapter.StepAdapter;
import ch.alpine.subare.core.td.Sarsa;
import ch.alpine.subare.core.td.SarsaType;
import ch.alpine.subare.demo.airport.Airport;
import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;

class UcbPolicyTest {
  @Test
  void testSimple() {
    Airport airport = Airport.INSTANCE;
    DiscreteQsa qsa = DiscreteQsa.build(airport);
    StateActionCounter sac = new DiscreteStateActionCounter();
    Sarsa sarsa = SarsaType.ORIGINAL.sarsa(airport, ConstantLearningRate.of(RealScalar.ZERO), //
        qsa, sac, PolicyType.EGREEDY.bestEquiprobable(airport, qsa, sac));
    for (Tensor state : airport.states()) {
      for (Tensor action : airport.actions(state)) {
        assertFalse(sarsa.sac().isEncountered(StateAction.key(state, action)));
        assertEquals(sarsa.sac().stateActionCount(StateAction.key(state, action)), RealScalar.ZERO);
        assertEquals(sarsa.sac().stateCount(state), RealScalar.ZERO);
      }
    }
    Tensor state = airport.states().get(0);
    Tensor action = airport.actions(state).get(0);
    Tensor nextState = airport.move(state, action);
    sarsa.digest(new StepAdapter(state, action, RealScalar.ZERO, nextState));
    for (Tensor s : airport.states()) {
      for (Tensor a : airport.actions(state)) {
        if (state.equals(s)) {
          assertEquals(sarsa.sac().stateCount(s), RealScalar.ONE);
          if (action.equals(a)) {
            assertTrue(sarsa.sac().isEncountered(StateAction.key(s, a)));
            assertEquals(sarsa.sac().stateActionCount(StateAction.key(s, a)), RealScalar.ONE);
          } else {
            assertEquals(sarsa.sac().stateActionCount(StateAction.key(s, a)), RealScalar.ZERO);
            assertFalse(sarsa.sac().isEncountered(StateAction.key(s, a)));
          }
        } else {
          assertEquals(sarsa.sac().stateCount(s), RealScalar.ZERO);
        }
      }
    }
  }

  @Test
  void testUcb() {
    Airport airport = Airport.INSTANCE;
    DiscreteQsa qsa = DiscreteQsa.build(airport);
    StateActionCounter sac = new DiscreteStateActionCounter();
    Sarsa sarsa = SarsaType.ORIGINAL.sarsa(airport, ConstantLearningRate.of(RealScalar.ZERO), //
        qsa, sac, PolicyType.EGREEDY.bestEquiprobable(airport, qsa, sac));
    DiscreteQsa ucbInQsa = UcbUtils.getUcbInQsa(airport, qsa, sarsa.sac());
    for (Tensor state : airport.states()) {
      for (Tensor action : airport.actions(state)) {
        assertEquals(UcbUtils.getUpperConfidenceBound(state, action, qsa.value(state, action), sarsa.sac()), DoubleScalar.POSITIVE_INFINITY);
        assertEquals(ucbInQsa.value(state, action), DoubleScalar.POSITIVE_INFINITY);
      }
    }
    Tensor state = airport.states().get(0);
    Tensor action = airport.actions(state).get(0);
    Tensor nextState = airport.move(state, action);
    sarsa.digest(new StepAdapter(state, action, RealScalar.ZERO, nextState));
    ucbInQsa = UcbUtils.getUcbInQsa(airport, qsa, sarsa.sac());
    for (Tensor s : airport.states()) {
      for (Tensor a : airport.actions(s)) {
        if (s.equals(state) && a.equals(action))
          assertEquals(ucbInQsa.value(s, a), RealScalar.ZERO);
        else
          assertEquals(ucbInQsa.value(s, a), DoubleScalar.POSITIVE_INFINITY);
      }
    }
  }
}
