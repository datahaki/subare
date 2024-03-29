// code by jph
package ch.alpine.subare.demo.virtualstations;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.red.Total;
import ch.alpine.tensor.sca.pow.Power;

class VirtualStationsTest {
  // each state the time interval followed by the NVnodes many virtual node informations
  // per time interval there are 2^NVnodes many different states (the end state is an additional interval)
  @Test
  void testStateSize() {
    VirtualStations virtualStations = (VirtualStations) VirtualStations.INSTANCE;
    virtualStations.states().forEach(v -> assertEquals(v.length(), virtualStations.getNVnodes() + 1));
    assertEquals(virtualStations.states().length(), (int) ((virtualStations.getTimeIntervals() + 1) * Math.pow(2, virtualStations.getNVnodes())));
  }

  // startStates all contain the lowest inveral number 0
  @Test
  void testStartState() {
    VirtualStations virtualStations = (VirtualStations) VirtualStations.INSTANCE;
    Tensor startStates = virtualStations.startStates();
    startStates.forEach(v -> assertEquals(v.Get(0), RealScalar.ZERO));
  }

  @Test
  void testActions() {
    VirtualStations virtualStations = (VirtualStations) VirtualStations.INSTANCE;
    for (Tensor state : virtualStations.states()) {
      Tensor actions = virtualStations.actions(state);
      if (virtualStations.isTerminal(state)) {
        assertEquals(actions.length(), 1);
      } else {
        Scalar expected = Power.of(RealScalar.of(2),
            Total.ofVector(state.extract(1, state.length())).multiply(RealScalar.of(virtualStations.getNVnodes() - 1)));
        Scalar actual = RealScalar.of(actions.length());
        assertEquals(expected, actual);
      }
    }
  }

  @Test
  void testTerminalStates() {
    VirtualStations virtualStations = (VirtualStations) VirtualStations.INSTANCE;
    for (Tensor state : virtualStations.states()) {
      assertEquals(virtualStations.isTerminal(state), state.Get(0).equals(RealScalar.of(virtualStations.getTimeIntervals())));
    }
  }
}
