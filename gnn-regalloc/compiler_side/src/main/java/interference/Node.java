package interference;

/** A virtual register (or source-level temporary) in an interference graph. */
public final class Node {
    private final int id;
    private int degree;
    private final int liveRangeLength;
    private final int loopDepth;
    private final int useDefCount;
    private boolean moveRelated;
    private Integer labelRegister;
    private boolean labelSpill;

    public Node(int id, int liveRangeLength, int loopDepth, int useDefCount) {
        if (liveRangeLength < 0 || loopDepth < 0 || useDefCount < 0) {
            throw new IllegalArgumentException("Node metrics must be non-negative");
        }
        this.id = id;
        this.liveRangeLength = liveRangeLength;
        this.loopDepth = loopDepth;
        this.useDefCount = useDefCount;
    }

    public int getId() { return id; }
    public int getDegree() { return degree; }
    public int getLiveRangeLength() { return liveRangeLength; }
    public int getLoopDepth() { return loopDepth; }
    public int getUseDefCount() { return useDefCount; }
    public boolean isMoveRelated() { return moveRelated; }
    public Integer getLabelRegister() { return labelRegister; }
    public boolean isLabelSpill() { return labelSpill; }

    void incrementDegree() { degree++; }
    void markMoveRelated() { moveRelated = true; }

    /** Sets the ground-truth allocation label; a null register denotes SPILL. */
    public void setAllocationLabel(Integer register) {
        if (register != null && register < 0) {
            throw new IllegalArgumentException("Register id must be non-negative");
        }
        labelRegister = register;
        labelSpill = register == null;
    }

    public void clearAllocationLabel() {
        labelRegister = null;
        labelSpill = false;
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof Node node && id == node.id;
    }

    @Override
    public int hashCode() { return Integer.hashCode(id); }

    @Override
    public String toString() { return "Node{" + id + "}"; }
}
