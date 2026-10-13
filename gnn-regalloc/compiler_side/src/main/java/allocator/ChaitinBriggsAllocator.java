package allocator;

import interference.Edge;
import interference.InterferenceGraph;
import interference.Node;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * Conservative Chaitin-Briggs allocator used as the dataset label generator.
 * Register IDs are zero-based: {@code 0 .. numRegisters - 1}; null means SPILL.
 */
public final class ChaitinBriggsAllocator {
    public AllocationResult allocate(InterferenceGraph graph, int numRegisters) {
        if (numRegisters < 1) throw new IllegalArgumentException("At least one register is required");

        List<Node> nodes = graph.getNodes().stream()
                .sorted(Comparator.comparingInt(Node::getId)).toList();
        Map<Integer, Integer> parent = new HashMap<>();
        for (Node node : nodes) {
            parent.put(node.getId(), node.getId());
            node.clearAllocationLabel();
        }

        Map<Edge, Boolean> coalesceSafety = coalesceConservatively(graph, numRegisters, parent);
        Map<Integer, Set<Integer>> adjacency = representativeAdjacency(graph, parent);
        Deque<StackEntry> stack = simplifyAndChooseSpills(graph, parent, adjacency, numRegisters);
        Map<Integer, Integer> colorsByRepresentative = selectColors(stack, adjacency, numRegisters);

        Map<Integer, Integer> registersByNode = new LinkedHashMap<>();
        Set<Integer> spilledNodeIds = new LinkedHashSet<>();
        for (Node node : nodes) {
            Integer color = colorsByRepresentative.get(find(parent, node.getId()));
            node.setAllocationLabel(color);
            if (color == null) spilledNodeIds.add(node.getId());
            else registersByNode.put(node.getId(), color);
        }
        return new AllocationResult(registersByNode, spilledNodeIds, coalesceSafety);
    }

    private Map<Edge, Boolean> coalesceConservatively(InterferenceGraph graph, int k,
                                                        Map<Integer, Integer> parent) {
        Map<Edge, Boolean> safety = new LinkedHashMap<>();
        for (Edge move : graph.getMoveEdges()) {
            int left = find(parent, move.getFirst().getId());
            int right = find(parent, move.getSecond().getId());
            if (left == right) {
                safety.put(move, true);
                continue;
            }
            if (groupsInterfere(graph, parent, left, right)) {
                safety.put(move, false);
                continue;
            }

            Map<Integer, Set<Integer>> adjacency = representativeAdjacency(graph, parent);
            Set<Integer> mergedNeighbors = new HashSet<>(adjacency.get(left));
            mergedNeighbors.addAll(adjacency.get(right));
            mergedNeighbors.remove(left);
            mergedNeighbors.remove(right);
            long significantNeighbors = mergedNeighbors.stream()
                    .filter(neighbor -> adjacency.get(neighbor).size() >= k)
                    .count();

            // Briggs: coalesce only if fewer than k neighbours are high-degree.
            boolean safe = significantNeighbors < k;
            safety.put(move, safe);
            if (safe) union(parent, left, right);
        }
        return safety;
    }

    private Deque<StackEntry> simplifyAndChooseSpills(InterferenceGraph graph,
                                                        Map<Integer, Integer> parent,
                                                        Map<Integer, Set<Integer>> adjacency, int k) {
        Set<Integer> active = new TreeSet<>(adjacency.keySet());
        Deque<StackEntry> stack = new ArrayDeque<>();
        while (!active.isEmpty()) {
            Integer simplifiable = active.stream()
                    .filter(node -> activeDegree(node, adjacency, active) < k)
                    .findFirst().orElse(null);
            if (simplifiable != null) {
                stack.push(new StackEntry(simplifiable, false));
                active.remove(simplifiable);
                continue;
            }

            int spillCandidate = active.stream()
                    .min((left, right) -> compareSpillCost(left, right, graph, parent, adjacency, active))
                    .orElseThrow();
            stack.push(new StackEntry(spillCandidate, true));
            active.remove(spillCandidate);
        }
        return stack;
    }

    private Map<Integer, Integer> selectColors(Deque<StackEntry> stack,
                                                 Map<Integer, Set<Integer>> adjacency, int k) {
        Map<Integer, Integer> colors = new HashMap<>();
        while (!stack.isEmpty()) {
            StackEntry entry = stack.pop();
            Set<Integer> forbidden = new HashSet<>();
            for (int neighbor : adjacency.get(entry.representative)) {
                Integer color = colors.get(neighbor);
                if (color != null) forbidden.add(color);
            }

            Integer selected = null;
            for (int candidate = 0; candidate < k; candidate++) {
                if (!forbidden.contains(candidate)) {
                    selected = candidate;
                    break;
                }
            }
            if (selected != null) colors.put(entry.representative, selected);
            // Otherwise this potential spill becomes an actual SPILL label.
        }
        return colors;
    }

    private int compareSpillCost(int left, int right, InterferenceGraph graph,
                                 Map<Integer, Integer> parent, Map<Integer, Set<Integer>> adjacency,
                                 Set<Integer> active) {
        double leftCost = spillCost(left, graph, parent, activeDegree(left, adjacency, active));
        double rightCost = spillCost(right, graph, parent, activeDegree(right, adjacency, active));
        int comparison = Double.compare(leftCost, rightCost);
        return comparison != 0 ? comparison : Integer.compare(left, right);
    }

    private double spillCost(int representative, InterferenceGraph graph,
                             Map<Integer, Integer> parent, int degree) {
        int useDefCount = graph.getNodes().stream()
                .filter(node -> find(parent, node.getId()) == representative)
                .mapToInt(Node::getUseDefCount).sum();
        return (double) useDefCount / degree;
    }

    private int activeDegree(int node, Map<Integer, Set<Integer>> adjacency, Set<Integer> active) {
        return (int) adjacency.get(node).stream().filter(active::contains).count();
    }

    private boolean groupsInterfere(InterferenceGraph graph, Map<Integer, Integer> parent,
                                    int left, int right) {
        for (Edge edge : graph.getEdges()) {
            int edgeLeft = find(parent, edge.getFirst().getId());
            int edgeRight = find(parent, edge.getSecond().getId());
            if ((edgeLeft == left && edgeRight == right) || (edgeLeft == right && edgeRight == left)) {
                return true;
            }
        }
        return false;
    }

    private Map<Integer, Set<Integer>> representativeAdjacency(InterferenceGraph graph,
                                                                 Map<Integer, Integer> parent) {
        Map<Integer, Set<Integer>> adjacency = new HashMap<>();
        for (Node node : graph.getNodes()) adjacency.putIfAbsent(find(parent, node.getId()), new TreeSet<>());
        for (Edge edge : graph.getEdges()) {
            int left = find(parent, edge.getFirst().getId());
            int right = find(parent, edge.getSecond().getId());
            if (left != right) {
                adjacency.get(left).add(right);
                adjacency.get(right).add(left);
            }
        }
        return adjacency;
    }

    private int find(Map<Integer, Integer> parent, int node) {
        int root = parent.get(node);
        if (root != node) {
            root = find(parent, root);
            parent.put(node, root);
        }
        return root;
    }

    private void union(Map<Integer, Integer> parent, int left, int right) {
        int leftRoot = find(parent, left);
        int rightRoot = find(parent, right);
        if (leftRoot == rightRoot) return;
        // Deterministic representative keeps labels reproducible under a fixed input graph.
        if (leftRoot < rightRoot) parent.put(rightRoot, leftRoot);
        else parent.put(leftRoot, rightRoot);
    }

    private record StackEntry(int representative, boolean potentialSpill) { }
}
