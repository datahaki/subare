// code by fluric
package ch.alpine.subare.util;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.RationalScalar;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.sca.Chop;

class CoinflipTest {
  @Test
  void testProbabilityDistribution() {
    Scalar headProbability0 = RealScalar.of(0.1);
    Scalar headProbability1 = RealScalar.of(0.5);
    Scalar headProbability2 = RealScalar.of(0.9);
    Coinflip coinflip0 = Coinflip.of(headProbability0);
    Coinflip coinflip1 = Coinflip.of(headProbability1);
    Coinflip coinflip2 = Coinflip.of(headProbability2);
    int[] counters = { 0, 0, 0 };
    int rounds = 100000;
    for (int i = 0; i < rounds; ++i) {
      counters[0] += coinflip0.tossHead() ? 1 : 0;
      counters[1] += coinflip1.tossHead() ? 1 : 0;
      counters[2] += coinflip2.tossHead() ? 1 : 0;
    }
    Chop._02.requireClose(RationalScalar.of(counters[0], rounds), headProbability0);
    Chop._02.requireClose(RationalScalar.of(counters[1], rounds), headProbability1);
    Chop._02.requireClose(RationalScalar.of(counters[2], rounds), headProbability2);
  }

  @Test
  void testInstances() {
    assertTrue(Coinflip.fair() != Coinflip.fair());
  }

  @Test
  void testFail() {
    assertThrows(Exception.class, () -> Coinflip.of(RealScalar.of(-0.1)));
    assertThrows(Exception.class, () -> Coinflip.of(RealScalar.of(1.1)));
  }
}
