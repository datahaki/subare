// code by jph
package ch.alpine.subare.net;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Stream;

import ch.alpine.subare.net.api.NetLayer;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.alg.Dimensions;
import ch.alpine.tensor.alg.Join;
import ch.alpine.tensor.alg.Transpose;

public enum NetExtract {
  ;
  public static Stream<NetLayer> flatten(NetLayer netLayer) {
    return Stream.concat( //
        Stream.of(netLayer), //
        netLayer.nested().stream().flatMap(layer -> flatten(layer)));
  }

  public static List<Tensor> parameters(NetLayer netLayer) {
    List<Tensor> list = new LinkedList<>();
    build(list, netLayer.parameters());
    return list;
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
        list.add(tensor.maps(Tensors::of));
        return;
      }
    } else
      tensor.forEach(element -> build(list, element));
  }

  public static Tensor parametersAsMatrix(NetLayer netLayer) {
    List<Tensor> list = parameters(netLayer).stream().map(m -> {
      List<Integer> dims = Dimensions.of(m);
      return dims.get(0) < dims.get(1) ? Transpose.of(m) : m;
    }).toList();
    List<Dimensions> dims = list.stream().map(Dimensions::new).toList();
    int max = dims.stream().mapToInt(d -> d.list().get(0)).max().orElse(0);
    List<Tensor> result = new ArrayList<>();
    for (int i = 0; i < list.size(); ++i) {
      Tensor mat = list.get(i);
      List<Integer> loc = dims.get(i).list();
      int act = loc.get(0);
      int dif = max - act;
      result.add(0 < dif ? Join.of(mat, Array.zeros(dif, loc.get(1))) : mat);
    }
    return Join.of(1, result);
  }
}
