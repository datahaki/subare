// code by jph
package ch.alpine.subare.ch05.blackjack;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.core.api.EpisodeInterface;
import ch.alpine.subare.core.api.Policy;
import ch.alpine.subare.core.util.EpisodeKickoff;
import ch.alpine.subare.core.util.EquiprobablePolicy;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.red.Tally;

class BlackjackTest {
  @SuppressWarnings("unused")
  @Test
  void testSimple() {
    Blackjack blackjack = new Blackjack();
    // TODO SUBARE fail sometimes, correct or wrong?
    {
      Tensor next = blackjack.move(Tensors.vector(0, 18, 7), RealScalar.ONE);
      // assertEquals(next, Tensors.vector(-1));
    }
    {
      Tensor next = blackjack.move(Tensors.vector(0, 21, 7), RealScalar.ZERO);
      // assertEquals(next, Tensors.vector(1));
    }
  }

  @Test
  void testEpisodeLength() {
    Blackjack blackjack = new Blackjack();
    Policy pi = EquiprobablePolicy.create(blackjack);
    Tensor tally = Tensors.empty();
    for (int episodes = 0; episodes < 10000; ++episodes) {
      EpisodeInterface ei = EpisodeKickoff.single(blackjack, pi);
      int count = 0;
      while (ei.hasNext()) {
        ei.step();
        ++count;
      }
      tally.append(RealScalar.of(count));
    }
    Map<Tensor, Long> map = Tally.of(tally);
    // {1=6574, 2=2537, 3=759, 4=121, 5=8, 7=1}
    // {2=2497, 1=6623, 6=1, 5=18, 4=138, 3=723}
    assertTrue(5 <= map.size());
    System.out.println("" + Tally.of(tally));
  }
}
