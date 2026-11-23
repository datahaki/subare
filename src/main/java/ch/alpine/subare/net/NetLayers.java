// code by jph
package ch.alpine.subare.net;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.random.RandomGenerator;
import java.util.stream.Stream;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.Unprotect;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.alg.Dimensions;
import ch.alpine.tensor.alg.Join;
import ch.alpine.tensor.alg.Transpose;
import ch.alpine.tensor.api.ScalarUnaryOperator;

public enum NetLayers {
  ;
  /** @param netLayer
   * @return */
  public static Stream<TrainableNetLayer> trainableLayers(NetLayer netLayer) {
    return NetExtract.flatten(netLayer) //
        .filter(TrainableNetLayer.class::isInstance) //
        .map(TrainableNetLayer.class::cast);
  }

  /** @param netLayer
   * @param randomGenerator */
  public static void initialize(NetLayer netLayer, RandomGenerator randomGenerator) {
    trainableLayers(netLayer).forEach(layer -> layer.initialize(randomGenerator));
  }

  /** @param netLayer */
  public static void initialize(NetLayer netLayer) {
    initialize(netLayer, ThreadLocalRandom.current());
  }

  /** @param netLayer
   * @return */
  public static Stream<Tensor> parameters(NetLayer netLayer) {
    return trainableLayers(netLayer).map(TrainableNetLayer::parameters);
  }

  /** @param netLayer
   * @param weightDecay
   * @return */
  public static Scalar reg_loss(NetLayer netLayer, boolean weightDecay) {
    return trainableLayers(netLayer) //
        .map(layer -> layer.reg_loss(weightDecay)) //
        .reduce(Scalar::add) //
        .orElse(RealScalar.ZERO);
  }

  /** @param netLayer
   * @return matrix */
  public static Tensor parametersAsMatrix(NetLayer netLayer) {
    List<Tensor> list2 = new LinkedList<>();
    build(list2, Tensor.of(parameters(netLayer)));
    List<Tensor> list = list2 //
        .stream() //
        .map(m -> {
          List<Integer> dims = Dimensions.of(m);
          return dims.get(0) > dims.get(1) ? Transpose.of(m) : m;
        }).toList();
    List<Dimensions> dims = list.stream().map(Dimensions::new).toList();
    int max = dims.stream().mapToInt(d -> d.list().get(1)).max().orElse(0);
    List<Tensor> result = new ArrayList<>();
    for (int i = 0; i < list.size(); ++i) {
      Tensor mat = list.get(i);
      List<Integer> loc = dims.get(i).list();
      int act = loc.get(1);
      int dif = max - act;
      result.add(0 < dif ? Join.of(1, mat, Array.zeros(loc.get(0), dif)) : mat);
    }
    return Join.of(0, result);
  }

  private static void build(List<Tensor> list, Tensor tensor) {
    Dimensions dimensions = new Dimensions(tensor);
    if (dimensions.isArray()) {
      if (dimensions.rank() == 2) {
        list.add(tensor);
        return;
      }
      List<Integer> dims = dimensions.list();
      if (dimensions.rank() == 1 && 0 < dims.get(0)) {
        // promote vector to row matrix
        list.add(Unprotect.byRef(tensor));
        return;
      }
    } else
      tensor.forEach(element -> build(list, element));
  }

  /** @param netLayer
   * @return */
  public static List<String> toString(NetLayer netLayer) {
    List<String> list = new LinkedList<>();
    recur(list, 0, netLayer);
    return list;
  }

  private static void recur(List<String> list, int level, NetLayer netLayer) {
    String indent = "| ".repeat(level);
    list.add(indent + netLayer);
    for (NetLayer layer : netLayer.nested().toList())
      recur(list, level + 1, layer);
  }

  public static ScalarUnaryOperator asSUO11(NetLayer netLayer) {
    return s -> netLayer.forward(Tensors.of(s)).Get(0);
  }

  public static NetOutput output(NetLayer netLayer) {
    return (NetOutput) NetExtract.flatten(netLayer) //
        .reduce((_, second) -> second) //
        .orElseThrow();
  }
}
