// code by jph
package ch.alpine.subare.util;

import java.io.Serializable;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.ext.Integers;

/** index is similar to a database index over the 0-level entries of the tensor.
 * the index allows fast checks for containment and gives the position of the key
 * in the original tensor of keys */
public class Index implements Serializable {
  /** @param tensor without duplicate entries
   * @return
   * @throws Exception if given tensor is a scalar
   * @throws Exception if tensor contains duplicate entries */
  public static Index build(Tensor tensor) {
    return new Index(tensor);
  }

  // ---
  private final Tensor keys;
  private final Map<Tensor, Integer> map;

  private Index(Tensor keys) {
    this.keys = keys;
    AtomicInteger atomicInteger = new AtomicInteger();
    map = keys.stream().collect(Collectors.toMap(k -> k, k -> atomicInteger.getAndIncrement()));
    Integers.requireEquals(keys.length(), map.size());
  }

  public Tensor keys() {
    return keys;
  }

  public Tensor get(int index) {
    return keys.get(index).unmodifiable();
  }

  public boolean containsKey(Tensor key) {
    return map.containsKey(key);
  }

  /** @param key
   * @return
   * @throws Exception if key does not exist in index */
  public int of(Tensor key) {
    return map.get(key);
  }

  public int size() {
    return keys.length();
  }
}
