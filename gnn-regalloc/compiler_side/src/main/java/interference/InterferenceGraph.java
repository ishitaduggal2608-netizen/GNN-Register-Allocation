package interference;

import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Undirected interference graph used as one sample in the allocation dataset.
 * Move edges are deliberately separate: they are candidates for coalescing,
 * not proof that the two values interfere.
 */
public final class InterferenceGraph {
    private final Map<Integer, Node> nodesById = new LinkedHashMap<>();
    private final Set<Edge> edges = new LinkedHashSet<>();
    private final Set<Edge> moveEdges = new LinkedHashSet<>();

    public Node addNode(int id, int liveRangeLength, int loopDepth, int useDefCount) {
        if (nodesById.containsKey(id)) {
            throw new IllegalArgumentException("Duplicate node id: " + id);
        }
        Node node = new Node(id, liveRangeLength, loopDepth, useDefCount);
        nodesById.put(id, node);
        return node;
    }

    public void addInterferenceEdge(int firstId, int secondId) {
        Node first = requireNode(firstId);
        Node second = requireNode(secondId);
        Edge edge = new Edge(first, second);
        if (edges.add(edge)) {
            first.incrementDegree();
            second.incrementDegree();
        }
    }

    public void addMoveEdge(int firstId, int secondId) {
        Node first = requireNode(firstId);
        Node second = requireNode(secondId);
        if (moveEdges.add(new Edge(first, second))) {
            first.markMoveRelated();
            second.markMoveRelated();
        }
    }

    public Node getNode(int id) { return requireNode(id); }
    public List<Node> getNodes() { return List.copyOf(nodesById.values()); }
    public List<Edge> getEdges() { return List.copyOf(edges); }
    public List<Edge> getMoveEdges() { return List.copyOf(moveEdges); }

    public boolean hasInterferenceEdge(int firstId, int secondId) {
        return edges.contains(new Edge(requireNode(firstId), requireNode(secondId)));
    }

    public Collection<Node> neighborsOf(int id) {
        Node node = requireNode(id);
        Set<Node> neighbors = new LinkedHashSet<>();
        for (Edge edge : edges) {
            if (edge.getFirst().equals(node)) neighbors.add(edge.getSecond());
            if (edge.getSecond().equals(node)) neighbors.add(edge.getFirst());
        }
        return Collections.unmodifiableSet(neighbors);
    }

    private Node requireNode(int id) {
        Node node = nodesById.get(id);
        if (node == null) {
            throw new IllegalArgumentException("Unknown node id: " + id);
        }
        return node;
    }
}
