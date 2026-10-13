package cfg;

import java.util.Objects;

/** Immutable name for a virtual register or temporary in three-address code. */
public final class Variable {
    private final String name;

    public Variable(String name) {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("Variable name must not be blank");
        }
        this.name = name;
    }

    public String getName() { return name; }

    @Override
    public boolean equals(Object other) {
        return other instanceof Variable variable && name.equals(variable.name);
    }

    @Override
    public int hashCode() { return name.hashCode(); }

    @Override
    public String toString() { return name; }
}
