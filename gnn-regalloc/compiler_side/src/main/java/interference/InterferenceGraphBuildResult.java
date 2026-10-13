package interference;

import cfg.Variable;

import java.util.Map;
import java.util.Objects;

/** Graph plus the source-variable-to-node mapping used to construct it. */
public final class InterferenceGraphBuildResult {
    private final InterferenceGraph graph;
    private final Map<Variable, Integer> nodeIdsByVariable;

    public InterferenceGraphBuildResult(InterferenceGraph graph, Map<Variable, Integer> nodeIdsByVariable) {
        this.graph = Objects.requireNonNull(graph, "graph");
        this.nodeIdsByVariable = Map.copyOf(nodeIdsByVariable);
    }

    public InterferenceGraph getGraph() { return graph; }
    public Map<Variable, Integer> getNodeIdsByVariable() { return nodeIdsByVariable; }
}
