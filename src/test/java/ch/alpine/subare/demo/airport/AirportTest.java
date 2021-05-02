// code by fluric
package ch.alpine.subare.demo.airport;

import java.util.stream.IntStream;

import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.TensorRuntimeException;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;
import junit.framework.TestCase;

public class AirportTest extends TestCase {
  public void testTerminalState() {
    Airport airport = Airport.INSTANCE;
    assertEquals(airport.isTerminal(Tensors.vector(Airport.LASTT, 0, Airport.VEHICLES)), true);
    assertEquals(airport.actions(Tensors.vector(Airport.LASTT, 0, Airport.VEHICLES)), Array.zeros(1, 1));
    assertEquals(airport.expectedReward(Tensors.vector(Airport.LASTT, 0, Airport.VEHICLES), Tensors.of(RealScalar.ZERO)), RealScalar.ZERO);
    assertEquals(
        airport.reward(Tensors.vector(Airport.LASTT, 0, Airport.VEHICLES), Tensors.of(RealScalar.ZERO), Tensors.vector(Airport.LASTT, 0, Airport.VEHICLES)),
        RealScalar.ZERO);
  }

  public void testCustProb() {
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
        throw TensorRuntimeException.of(state, action, R, mean);
        // fail(); // does not always work
      }
    }
  }
}
