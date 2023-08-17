// code by jph
package ch.alpine.subare.ch05.racetrack;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ch.alpine.subare.core.api.MonteCarloInterface;
import ch.alpine.subare.core.util.DeterministicStandardModel;
import ch.alpine.subare.core.util.StateActionMap;
import ch.alpine.subare.core.util.StateActionMaps;
import ch.alpine.subare.util.Index;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.alg.Dimensions;
import ch.alpine.tensor.alg.Flatten;
import ch.alpine.tensor.alg.Join;
import ch.alpine.tensor.alg.Subdivide;
import ch.alpine.tensor.itp.Interpolation;
import ch.alpine.tensor.itp.NearestInterpolation;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;

/** Exercise 5.8 p.111: Racetrack (programming)
 * Figure 5.6
 * 
 * the book states that the velocity components should be non-negative
 * the track layout however encourages nudging in the negative direction
 * so we make a compromise by using the following integration procedure
 * p' = p + v + a
 * v' = clip(v + a)
 * 
 * References:
 * Barto, Bradtke, and Singh (1995)
 * Gardner (1973) */
class Racetrack extends DeterministicStandardModel implements MonteCarloInterface {
  static final Tensor WHITE = Tensors.vector(255, 255, 255, 255);
  static final Tensor RED = Tensors.vector(255, 0, 0, 255);
  static final Tensor GREEN = Tensors.vector(0, 255, 0, 255);
  static final Tensor BLACK = Tensors.vector(0, 0, 0, 255);
  static final Scalar MINUS_ONE = RealScalar.ONE.negate();
  // ---
  private static final Tensor STATE_COLLISION = Tensors.vector(9999);
  // ---
  private final Clip clipPositionX;
  private final Clip clipPositionY;
  private final Clip clipSpeed;
  private final Tensor states = Tensors.empty(); // (px, py, vx, vy)
  final Tensor statesStart = Tensors.empty();
  final Tensor statesTerminal = Tensors.empty();
  final Tensor actions = //
      Flatten.of(Array.of(Tensors::vector, 3, 3), 1).map(s -> s.subtract(RealScalar.ONE)).unmodifiable();
  final Index statesIndex;
  final Index statesStartIndex;
  final Index statesTerminalIndex;
  private final StateActionMap stateActionMap;
  private final Interpolation interpolation;
  /** memo map is populated and reused in {@link #move(Tensor, Tensor)} */
  private final Map<Tensor, Boolean> memo_freeSpace = new HashMap<>();
  private final Tensor image;
  private final Tensor actionsTerminal = Tensors.vector(0); // do nothing

  Racetrack(Tensor image, int maxSpeed) {
    Tensor blue = image.get(Tensor.ALL, Tensor.ALL, 2);
    System.out.println("grid size=" + Dimensions.of(blue));
    interpolation = NearestInterpolation.of(blue);
    List<Integer> list = Dimensions.of(image);
    Tensor dimensions = Tensors.vector(list.get(0), list.get(1)).map(s -> s.subtract(RealScalar.ONE));
    clipPositionX = Clips.positive(dimensions.Get(0));
    clipPositionY = Clips.positive(dimensions.Get(1));
    clipSpeed = Clips.positive(maxSpeed);
    for (int y = 0; y < list.get(1); ++y)
      for (int x = 0; x < list.get(0); ++x) {
        final Tensor rgba = image.get(x, y).unmodifiable();
        if (!rgba.equals(WHITE)) {
          final Tensor pstate = Tensors.vector(x, y);
          if (rgba.equals(BLACK))
            for (int vx = 0; vx <= maxSpeed; ++vx)
              for (int vy = 0; vy <= maxSpeed; ++vy)
                if (vx != 0 || vy != 0)
                  states.append(Join.of(pstate, Tensors.vector(vx, vy)));
          // ---
          if (rgba.equals(GREEN)) {
            Tensor state = Join.of(pstate, Array.zeros(2));
            states.append(state);
            statesStart.append(state);
          }
          // ---
          if (rgba.equals(RED))
            for (int vx = 0; vx <= maxSpeed; ++vx)
              for (int vy = 0; vy <= maxSpeed; ++vy)
                if (vx != 0 || vy != 0) {
                  Tensor state = Join.of(pstate, Tensors.vector(vx, vy));
                  states.append(state);
                  statesTerminal.append(state);
                }
        }
      }
    states.append(STATE_COLLISION);
    statesTerminal.append(STATE_COLLISION);
    // ---
    statesIndex = Index.build(states);
    statesStartIndex = Index.build(statesStart);
    statesTerminalIndex = Index.build(statesTerminal);
    stateActionMap = StateActionMaps.build(states, actions, this);
    this.image = image;
  }

  @Override
  public Tensor states() {
    return states.unmodifiable();
  }

  @Override
  public Tensor actions(Tensor state) {
    return isTerminal(state) //
        ? actionsTerminal
        : stateActionMap.actions(state);
  }

  /** @param state of the form {px, py, vx, vy}
   * @param action of the form {ax, ay}
   * @return */
  private static Tensor shift(Tensor state, Tensor action) {
    Tensor pos = state.extract(0, 2);
    Tensor vel = state.extract(2, 4);
    vel = vel.add(action);
    return Join.of(pos.add(vel), vel);
  }

  /** @param state of the form {px, py, vx, vy}
   * @param action of the form {ax, ay}
   * @return */
  Tensor integrate(Tensor state, Tensor action) {
    Tensor next = shift(state, action); // add velocity
    next.set(clipPositionX, 0);
    next.set(clipPositionY, 1);
    next.set(clipSpeed, 2); // vx
    next.set(clipSpeed, 3); // vy
    return next;
  }

  @Override // from DiscreteModel
  public Scalar gamma() {
    return RealScalar.ONE;
  }

  // ---
  private static Tensor split(Tensor p, Tensor q, Scalar scalar) {
    return q.subtract(p).multiply(scalar).add(p);
  }

  @Override // from MoveInterface
  public Tensor move(Tensor state, Tensor action) {
    if (isTerminal(state))
      return state;
    Tensor next = integrate(state, action); // vy
    if (statesIndex.containsKey(next)) { // proper move
      Tensor pos0 = state.extract(0, 2);
      Tensor pos1 = next.extract(0, 2);
      Tensor key = Tensors.of(pos0, pos1);
      return memo_freeSpace.computeIfAbsent(key, i -> Subdivide.of(0, 1, 5).stream() //
          .map(Scalar.class::cast).map(lambda -> interpolation.Get(split(pos0, pos1, lambda))) //
          .allMatch(Scalars::isZero)) //
              ? next
              : STATE_COLLISION;
    }
    return STATE_COLLISION;
  }

  // boolean
  @Override // from RewardInterface
  public Scalar reward(Tensor state, Tensor action, Tensor next) {
    if (!isTerminal(state) && isTerminal(next))
      if (next.equals(STATE_COLLISION))
        return RealScalar.of(-10); // cost of collision, required for value iteration
    if (isTerminal(next))
      return RealScalar.ZERO;
    // if (integrate(state, action).equals(next))
    return MINUS_ONE;
  }

  boolean isStart(Tensor state) {
    return statesStartIndex.containsKey(state);
  }

  // ---
  @Override // from MonteCarloInterface
  public Tensor startStates() {
    return statesStart;
  }

  @Override // from TerminalInterface
  public boolean isTerminal(Tensor state) {
    return statesTerminalIndex.containsKey(state);
  }

  // ---
  public Tensor image() {
    return image.copy();
  }
}
