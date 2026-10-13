package cfg;

import java.util.Objects;
import java.util.Set;

/** A TAC instruction with explicitly supplied use and definition sets. */
public final class Instruction {
    private final int id;
    private final String text;
    private final Set<Variable> uses;
    private final Set<Variable> defs;
    private final boolean moveInstruction;

    public Instruction(int id, String text, Set<Variable> uses, Set<Variable> defs) {
        this(id, text, uses, defs, false);
    }

    public Instruction(int id, String text, Set<Variable> uses, Set<Variable> defs,
                       boolean moveInstruction) {
        this.id = id;
        this.text = Objects.requireNonNull(text, "text");
        this.uses = Set.copyOf(Objects.requireNonNull(uses, "uses"));
        this.defs = Set.copyOf(Objects.requireNonNull(defs, "defs"));
        if (moveInstruction && (this.uses.size() != 1 || this.defs.size() != 1)) {
            throw new IllegalArgumentException("A move instruction must have one use and one definition");
        }
        this.moveInstruction = moveInstruction;
    }

    public int getId() { return id; }
    public String getText() { return text; }
    public Set<Variable> uses() { return uses; }
    public Set<Variable> defs() { return defs; }
    public boolean isMoveInstruction() { return moveInstruction; }

    @Override
    public boolean equals(Object other) {
        return other instanceof Instruction instruction && id == instruction.id;
    }

    @Override
    public int hashCode() { return Integer.hashCode(id); }

    @Override
    public String toString() { return id + ": " + text; }
}
