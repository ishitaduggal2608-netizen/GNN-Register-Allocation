package exporter;

import allocator.AllocationResult;
import interference.Edge;
import interference.InterferenceGraph;
import interference.Node;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;

/** Writes one fully labelled graph in the JSON schema consumed by the Python side. */
public final class GraphJsonExporter {
    public String toJson(String graphId, int numRegisters, InterferenceGraph graph,
                         AllocationResult allocation) {
        if (graphId == null || graphId.isBlank()) throw new IllegalArgumentException("graphId is required");
        if (numRegisters < 1) throw new IllegalArgumentException("numRegisters must be positive");

        StringBuilder json = new StringBuilder();
        json.append("{\n  \"graph_id\": \"").append(escape(graphId)).append("\",");
        json.append("\n  \"num_registers\": ").append(numRegisters).append(',');
        json.append("\n  \"nodes\": [");
        List<Node> nodes = graph.getNodes().stream().sorted(Comparator.comparingInt(Node::getId)).toList();
        for (int i = 0; i < nodes.size(); i++) {
            Node node = nodes.get(i);
            if (node.getLabelRegister() == null && !node.isLabelSpill()) {
                throw new IllegalStateException("Allocate the graph before exporting it");
            }
            if (i > 0) json.append(',');
            json.append("\n    {")
                    .append("\n      \"id\": ").append(node.getId()).append(',')
                    .append("\n      \"degree\": ").append(node.getDegree()).append(',')
                    .append("\n      \"live_range_length\": ").append(node.getLiveRangeLength()).append(',')
                    .append("\n      \"loop_depth\": ").append(node.getLoopDepth()).append(',')
                    .append("\n      \"use_def_count\": ").append(node.getUseDefCount()).append(',')
                    .append("\n      \"is_move_related\": ").append(node.isMoveRelated()).append(',')
                    .append("\n      \"label_register\": ")
                    .append(node.getLabelRegister() == null ? "null" : node.getLabelRegister()).append(',')
                    .append("\n      \"label_spill\": ").append(node.isLabelSpill())
                    .append("\n    }");
        }
        json.append("\n  ],");

        List<Edge> interferenceEdges = sortedEdges(graph.getEdges());
        json.append("\n  \"interference_edges\": [");
        for (int i = 0; i < interferenceEdges.size(); i++) {
            if (i > 0) json.append(',');
            appendPair(json, interferenceEdges.get(i));
        }
        json.append("],");

        List<Edge> moveEdges = sortedEdges(graph.getMoveEdges());
        json.append("\n  \"move_edges\": [");
        for (int i = 0; i < moveEdges.size(); i++) {
            Edge edge = moveEdges.get(i);
            if (i > 0) json.append(',');
            json.append("\n    { \"pair\": ");
            appendPair(json, edge);
            json.append(", \"coalesce_safe\": ").append(allocation.isCoalesceSafe(edge)).append(" }");
        }
        json.append("\n  ]\n}");
        return json.toString();
    }

    public void write(Path destination, String graphId, int numRegisters,
                      InterferenceGraph graph, AllocationResult allocation) throws IOException {
        Path parent = destination.toAbsolutePath().getParent();
        if (parent != null) Files.createDirectories(parent);
        Files.writeString(destination, toJson(graphId, numRegisters, graph, allocation), StandardCharsets.UTF_8);
    }

    private List<Edge> sortedEdges(List<Edge> edges) {
        return edges.stream().sorted(Comparator
                .comparingInt((Edge edge) -> Math.min(edge.getFirst().getId(), edge.getSecond().getId()))
                .thenComparingInt(edge -> Math.max(edge.getFirst().getId(), edge.getSecond().getId()))).toList();
    }

    private void appendPair(StringBuilder json, Edge edge) {
        int first = Math.min(edge.getFirst().getId(), edge.getSecond().getId());
        int second = Math.max(edge.getFirst().getId(), edge.getSecond().getId());
        json.append('[').append(first).append(", ").append(second).append(']');
    }

    private String escape(String value) {
        return value.replace("\\", "\\\\").replace("\"", "\\\"")
                .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t");
    }
}
