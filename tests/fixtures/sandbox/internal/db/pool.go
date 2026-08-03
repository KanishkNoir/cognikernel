package db

import (
	"context"
	"database/sql"
)

type (
	Pool struct {
		dsn      string
		maxConns int
	}
	Store interface {
		Get(ctx context.Context, id int) (*Row, error)
		Save(ctx context.Context, r *Row) error
	}
)

type Row struct {
	ID     int
	Amount int64
}

func NewPool(dsn string, maxConns int) *Pool {
	return &Pool{dsn: dsn, maxConns: maxConns}
}

func (p *Pool) Acquire(ctx context.Context) (*sql.Conn, error) {
	return nil, nil
}

func (p Pool) Stats() (int, int) { return 0, p.maxConns }
