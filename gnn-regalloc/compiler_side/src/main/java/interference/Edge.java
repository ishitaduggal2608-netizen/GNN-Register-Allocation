package interference;

import java.util.Objects;

/** An undirected relationship between two distinct graph nodes. */
public final class Edge {
    private final Node first;
    private final Node second;

    public Edge(Node first, Node second) {
        this.first = Objects.requireNonNull(first, "first");
        this.second = Objects.requireNonNull(second, "second");
        if (first.equals(second)) {
            throw new IllegalArgumentException("Self edges are not allowed");
        }
    }

    public Node getFirst() { return first; }
    public Node getSecond() { return second; }

    @Override
    public boolean equals(Object other) {
        if (!(other instanceof Edge edge)) {
            return false;
        }
        return (first.equals(edge.first) && second.equals(edge.second))
                || (first.equals(edge.second) && second.equals(edge.first));
    }

    @Override
    public int hashCode() { return first.hashCode() ^ second.hashCode(); }
}
