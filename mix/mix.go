package mix

import (
	"math/rand"

	"github.com/tsukanov-as/watf"
	"github.com/tsukanov-as/watf/internal"
)

// WIP

type number = internal.Number

type Mix[T number] struct {
	features int
	clusters int
	levels   int
	totals   []T
	stat     []T
	mean     []T
	watf     []*watf.Watf[T]
}

func New[T number](classes, features, levels int) *Mix[T] {
	clusters := (1 << (levels + 1)) - 2
	mix := &Mix[T]{
		levels:   levels,
		features: features,
		clusters: clusters,
		totals:   make([]T, clusters),
		stat:     make([]T, clusters*features),
		mean:     make([]T, clusters*features),
		watf:     make([]*watf.Watf[T], 1<<levels),
	}
	for i := range mix.watf {
		mix.watf[i] = watf.New[T](classes, features)
	}

	return mix
}

func (mix *Mix[T]) Feed(x []T, cluster int) {
	mix.totals[cluster] += 1
	s := mix.stat[cluster*mix.features:]
	for i, v := range x {
		s[i] += v
	}
}

func (mix *Mix[T]) FeedRandom(x []T) {
	c := rand.Intn(mix.clusters)
	mix.totals[c] += 1
	s := mix.stat[c*mix.features:]
	for i, v := range x {
		s[i] += v
	}
}

func (mix *Mix[T]) Commit() {
	base := 0
	for c := 0; c < mix.clusters; c++ {
		t := mix.totals[c]
		s := mix.stat[base:]
		m := mix.mean[base:]
		for i := 0; i < mix.features; i++ {
			m[i] = s[i] / t
		}
		base += mix.features
	}
}

func (mix *Mix[T]) Predict(x []T) int {
	return mix.Shard(x).Predict(x)
}

func (mix *Mix[T]) Shard(x []T) *watf.Watf[T] {
	return mix.watf[mix.clusters-mix.Cluster(x, mix.levels-1)-1]
}

func (mix *Mix[T]) Cluster(x []T, level int) int {
	cluster := 0
	last := 0
	base := 0
	for i := 0; i <= level; i++ {
		lh := diff(x, mix.mean[base:])
		rh := diff(x, mix.mean[base+mix.features:])
		if lh > rh {
			cluster += 1 // second cluster won
		}
		last = cluster
		cluster = (cluster + 1) * 2 // first subcluster of the winner
		base = cluster * mix.features
	}
	return last
}

func diff[T number](a, b []T) T {
	var sum T
	for i, v := range a {
		delta := b[i] - v
		sum += delta * delta
	}
	return sum
}
