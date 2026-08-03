package com.acme.billing;

import java.util.List;
import java.math.BigDecimal;

@Entity
public class Invoice implements Comparable<Invoice> {
    private BigDecimal total;
    private List<LineItem> items;

    @Override
    public int compareTo(Invoice other) { return 0; }

    public BigDecimal getTotal() { return total; }

    public static class Builder {
        public Invoice build() { return new Invoice(); }
    }
}

public interface LineItem { BigDecimal amount(); }
