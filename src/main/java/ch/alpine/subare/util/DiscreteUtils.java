// code by jph
package ch.alpine.subare.util;

import java.util.function.BinaryOperator;
import java.util.function.Function;

import ch.alpine.subare.api.mod.DiscreteModel;
import ch.alpine.subare.api.mod.StandardModel;
import ch.alpine.subare.api.val.QsaInterface;
import ch.alpine.subare.api.val.VsInterface;
import ch.alpine.subare.math.Index;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.api.ScalarUnaryOperator;
import ch.alpine.tensor.red.Max;

public enum DiscreteUtils {
  ;
  /** collects all possible state-action pairs that can be built with {@code states}
   * 
   * @param discreteModel
   * @param states
   * @return index for state-action */
  public static Index build(DiscreteModel discreteModel, Tensor states) {
    Tensor tensor = Tensors.empty();
    for (Tensor state : states)
      for (Tensor action : discreteModel.actions(state))
        tensor.append(StateAction.key(state, action));
    return Index.build(tensor);
  }

  // ---
  /** @param discreteModel
   * @param qsa
   * @param binaryOperator
   * @return */
  public static DiscreteVs reduce( //
      DiscreteModel discreteModel, QsaInterface qsa, BinaryOperator<Scalar> binaryOperator) {
    return DiscreteVs.build(discreteModel.states(), //
        Tensor.of(discreteModel.states().stream() //
            .map(state -> discreteModel.actions(state).stream() //
                .map(action -> qsa.value(state, action)) //
                .reduce(binaryOperator).orElseThrow()))); // <- assumes greedy policy
  }

  /** compute state value function v(s) based on given action-value function q(s, a)
   * 
   * @param discreteModel
   * @param qsa
   * @return state values */
  public static DiscreteVs createVs(DiscreteModel discreteModel, QsaInterface qsa) {
    return reduce(discreteModel, qsa, Max::of);
  }

  // ---
  public static void print(DiscreteQsa qsa, Function<Scalar, Scalar> round) {
    for (Tensor key : qsa.keys()) {
      Scalar value = qsa.value(key.get(0), key.get(1));
      System.out.println(key + " " + value.maps(round));
    }
  }

  public static void print(DiscreteQsa qsa) {
    print(qsa, Function.identity());
  }

  // ---
  public static String infoString(DiscreteQsa qsa) {
    StringBuilder stringBuilder = new StringBuilder();
    stringBuilder.append("#{q(s,a)}=").append(qsa.size()).append("\n");
    stringBuilder.append("   min(q)=").append(qsa.getMin()).append("\n");
    stringBuilder.append("   max(q)=").append(qsa.getMax()).append("\n");
    return stringBuilder.toString().trim();
  }

  // ---
  public static void print(DiscreteModel discreteModel, VsInterface vs, Function<Scalar, Scalar> round) {
    for (Tensor key : discreteModel.states()) {
      Scalar value = vs.value(key);
      System.out.println(key + " " + value.maps(round));
    }
  }

  public static void print(DiscreteVs vs, ScalarUnaryOperator round) {
    for (Tensor key : vs.keys()) {
      Scalar value = vs.value(key);
      System.out.println(key + " " + value.maps(round));
    }
  }

  public static void print(DiscreteVs vs) {
    print(vs, scalar -> scalar);
  }

  /** @param standardModel
   * @param vs
   * @return */
  public static QsaInterface getQsaFromVs(StandardModel standardModel, VsInterface vs) {
    ActionValueAdapter actionValueAdapter = new ActionValueAdapter(standardModel);
    DiscreteQsa qsa = DiscreteQsa.build(standardModel);
    for (Tensor state : standardModel.states())
      for (Tensor action : standardModel.actions(state))
        qsa.assign(state, action, actionValueAdapter.qsa(state, action, vs));
    return qsa;
  }
}
