// code by jph
package ch.alpine.subare.book.ch02;

import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.ext.Integers;

public abstract class Agent {
  protected static final RandomGenerator RANDOM = new SecureRandom();
  // ---
  private Scalar total = RealScalar.ZERO;
  private Integer count = 0;
  private int count_copy;
  private int randomizedDecisionCount = 0;
  private int actionReminder;
  private final Tensor actions = Tensors.empty();
  private final Tensor qvalues = Tensors.empty();
  /** EXPERIMENTAL SUBARE opening sequence of action */
  protected final List<Integer> openingSequence = new ArrayList<>();

  protected abstract int protected_takeAction();

  public final int takeAction() {
    actionReminder = protected_takeAction();
    return actionReminder;
  }

  public final int getActionReminder() {
    return actionReminder;
  }

  // shall only be used to recording history
  protected abstract Tensor protected_QValues();

  // shall not call getCount() from within
  protected abstract void protected_feedback(int a, Scalar value);

  public final void feedback(int a, Scalar value) {
    Integers.requireEquals(a, actionReminder);
    total = total.add(value);
    actions.append(RealScalar.of(a));
    ++count;
    count_copy = count;
    count = null; // prevent functions to use getCount()
    qvalues.append(protected_QValues());
    protected_feedback(a, value);
    count = count_copy;
  }

  public final void notifyAboutRandomizedDecision() {
    ++randomizedDecisionCount;
  }

  public final Scalar getCount() {
    return RealScalar.of(count);
  }

  public final Scalar getRewardTotal() {
    return total;
  }

  public final Scalar getRewardAverage() {
    return total.divide(RealScalar.of(count_copy));
  }

  public final Tensor getActions() {
    return actions.unmodifiable();
  }

  public final Tensor getQValues() {
    return qvalues.unmodifiable();
  }

  public abstract String getDescription();

  public final String getAbsDesc() {
    String name = getClass().getSimpleName();
    name = name.substring(0, name.length() - 5); // drop "Agent"
    return String.format("%10s%15s", name, getDescription());
  }

  @Override
  public final String toString() {
    return getAbsDesc();
  }

  public final int getRandomizedDecisionCount() {
    return randomizedDecisionCount;
  }

  public final void setOpeningSequence(Integer... actions) {
    // TODO SUBARE set != addAll !?
    openingSequence.addAll(List.of(actions));
  }
}
