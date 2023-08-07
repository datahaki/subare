// code by fluric
package ch.alpine.subare.demo.airport;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.stream.IntStream;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.core.StateActionCounter;
import ch.alpine.subare.core.adapter.StepAdapter;
import ch.alpine.subare.core.td.Sarsa;
import ch.alpine.subare.core.td.SarsaType;
import ch.alpine.subare.core.util.ConstantLearningRate;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.subare.core.util.DiscreteStateActionCounter;
import ch.alpine.subare.core.util.PolicyType;
import ch.alpine.subare.core.util.StateAction;
import ch.alpine.subare.core.util.UcbUtils;
import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.Throw;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;

class AirportTest {
  @Test
  void testTerminalState() {
    Airport airport = Airport.INSTANCE;
    assertTrue(airport.isTerminal(Tensors.vector(Airport.LASTT, 0, Airport.VEHICLES)));
    assertEquals(airport.actions(Tensors.vector(Airport.LASTT, 0, Airport.VEHICLES)), Array.zeros(1, 1));
    assertEquals(airport.expectedReward(Tensors.vector(Airport.LASTT, 0, Airport.VEHICLES), Tensors.of(RealScalar.ZERO)), RealScalar.ZERO);
    assertEquals(
        airport.reward(Tensors.vector(Airport.LASTT, 0, Airport.VEHICLES), Tensors.of(RealScalar.ZERO), Tensors.vector(Airport.LASTT, 0, Airport.VEHICLES)),
        RealScalar.ZERO);
  }

  @Test
  void testCustProb() {
    Airport airport = Airport.INSTANCE;
    Tensor state = Tensors.vector(1, 2, 3);
    Tensor actions = airport.actions(state);
    assertEquals(actions.length(), 12);
    int probes = 3000;
    Clip clip = Clips.absolute(2);
    for (Tensor action : actions) {
      Tensor next = airport.move(state, action);
      assertEquals(next.get(0), RealScalar.of(2));
      Scalar R = airport.expectedReward(state, action);
      Scalar total = IntStream.range(0, probes) //
          .mapToObj(i -> airport.reward(state, action, next)) //
          .reduce(Scalar::add).get();
      Scalar mean = total.divide(DoubleScalar.of(probes));
      if (!clip.isInside(R.subtract(mean))) {
        System.err.println(state + " " + action);
        System.err.println(R + " " + mean);
        throw new Throw(state, action, R, mean);
        // fail(); // does not always work
      }
    }
  }

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
