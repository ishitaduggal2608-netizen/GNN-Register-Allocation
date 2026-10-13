package allocator;

import interference.Edge;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.OptionalInt;
import java.util.Set;

/** Ground-truth labels emitted by the classical register allocator. */
public final class AllocationResult {
    private final Map<Integer, Integer> registersByNode;
    private final Set<Integer> spilledNodeIds;
    private final Map<Edge, Boolean> coalesceSafety;

    AllocationResult(Map<Integer, Integer> registersByNode, Set<Integer> spilledNodeIds,
                     Map<Edge, Boolean> coalesceSafety) {
        this.registersByNode = Collections.unmodifiableMap(new LinkedHashMap<>(registersByNode));
        this.spilledNodeIds = Collections.unmodifiableSet(new LinkedHashSet<>(spilledNodeIds));
        this.coalesceSafety = Collections.unmodifiableMap(new LinkedHashMap<>(coalesceSafety));
    }

    public OptionalInt registerOf(int nodeId) {
        Integer register = registersByNode.get(nodeId);
        return register == null ? OptionalInt.empty() : OptionalInt.of(register);
    }

    public boolean isSpilled(int nodeId) { return spilledNodeIds.contains(nodeId); }
    public boolean isCoalesceSafe(Edge moveEdge) { return coalesceSafety.getOrDefault(moveEdge, false); }
    public Map<Edge, Boolean> getCoalesceSafety() { return coalesceSafety; }
}
